#!/usr/bin/env python3

import json
import tempfile
from collections import Counter

import caffe2.python.hypothesis_test_util as hu
import caffe2.python.predictor.predictor_exporter as pe
import hypothesis.strategies as st
import numpy as np
import torch
import torch.nn.functional as F
from caffe2.python import workspace
from hypothesis import given
from pytext.common.constants import DatasetFieldName, PredictorInputNames
from pytext.config import config_from_json
from pytext.config.component import create_exporter, create_model
from pytext.data import CommonMetadata
from pytext.data.joint_data_handler import SEQ_LENS
from pytext.fields import FieldMeta
from pytext.jobspec import (
    DocClassificationJobSpec,
    JointTextJobSpec,
    WordTaggingJobSpec,
)
from pytext.utils.onnx_utils import CAFFE2_DB_TYPE
from torchtext.vocab import Vocab


JOINT_CONFIG = """
{
    "model": {
        "representation": {
            "JointBLSTMRepresentation": {
                "lstm": {
                  "lstm_dim": 30,
                  "num_layers": 1
                }
            }
        },
        "decoder": {
            "use_doc_probs_in_word": true
        },
        "output_config": {
            "doc_output": {
              "loss": {
                "CrossEntropyLoss": {}
              }
            },
            "word_output": {
              "CRFOutputLayer": {}
            }
        }
    },
    "features": {
      "word_feat": {},
      "dict_feat": {}
    },
    "exporter": {

    }
}
"""

DOC_CONFIGS = [
    """
{
  "model": {
    "representation": {
        "BiLSTMPooling": {
        "pooling": {
          "MaxPool": {}
        }
      }
    },
    "output_config": {
        "loss": {
        "CrossEntropyLoss": {}
        }
    }
  },
  "features": {
    "dict_feat": {
        "embed_dim": 10
    }
  },
  "trainer": {
    "epochs": 1
  },
  "exporter": {

  }
}
""",
    """
{
  "model": {
      "representation": {
        "DocNNRepresentation": {}
      },
      "output_config": {
        "loss": {
            "CrossEntropyLoss": {}
            }
      }
  },
  "features": {
    "word_feat": {},
    "dict_feat": {

    }
  },
  "trainer": {
    "epochs": 1
  },
  "exporter": {

  }
}

    """,
]
WORD_CONFIGS = [
    """{
    "model": {
        "representation": {
            "BiLSTMSlotAttention": {
                "lstm": {
                  "lstm_dim": 30,
                  "num_layers": 2
                }
            }
        },
        "output_config": {
            "WordTaggingOutputLayer": {}
        }
    },
    "features": {
        "dict_feat": {
            "embed_dim": 10
        }
    },
    "exporter": {}
  }
""",
    """{
    "model": {
        "representation": {
            "BiLSTMSlotAttention": {
                "lstm": {
                  "lstm_dim": 30,
                  "num_layers": 2
                }
            }
        },
        "output_config": {
            "CRFOutputLayer": {}
        }
    },
    "labels": {
      "word_label": {}
    },
    "features": {
      "word_feat": {},
      "dict_feat": {
        "embed_dim": 10
      }
    },
    "exporter": {}
  }
""",
]

W_VOCAB_SIZE = 10
UNK_IDX = 0
PAD_IDX = 1
W_VOCAB = ["<UNK>", "W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8", "W9"]
DICT_VOCAB_SIZE = 10
DICT_VOCAB = ["<UNK>", "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9"]
# For now we need to fix the batch_size for exporting and testing,
# Need to remove this and make it a random input once ONNX is able to
# Handle different batch_sizes
BATCH_SIZE = 1
INPUT_NAMES = [
    PredictorInputNames.TOKENS_IDS,
    PredictorInputNames.TOKENS_LENS,
    PredictorInputNames.DICT_FEAT_IDS,
    PredictorInputNames.DICT_FEAT_WEIGHTS,
    PredictorInputNames.DICT_FEAT_LENS,
]


class TextModelExporterTest(hu.HypothesisTestCase):
    @given(
        export_num_words=st.integers(1, 5),
        export_num_dict_feat=st.integers(1, 6),
        num_doc_classes=st.integers(2, 5),
        test_num_words=st.integers(1, 7),
        test_num_dict_feat=st.integers(1, 8),
        num_predictions=st.integers(1, 4),
    )
    def test_doc_export_to_caffe2(
        self,
        export_num_words,
        export_num_dict_feat,
        num_doc_classes,
        test_num_words,
        test_num_dict_feat,
        num_predictions,
    ):
        for config in DOC_CONFIGS:
            config = self._get_config(DocClassificationJobSpec, config)
            metadata = self._get_metadata(num_doc_classes, 0)
            py_model = create_model(config.model, config.features, metadata)
            exporter = create_exporter(
                config.exporter, config.features, config.labels, metadata
            )

            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".predictor"
            ) as pred_file:
                print(pred_file.name)
                output_names = exporter.export_to_caffe2(py_model, pred_file.name)
                workspace.ResetWorkspace()
            pred_net = pe.prepare_prediction_net(pred_file.name, CAFFE2_DB_TYPE)

            for _i in range(num_predictions):
                pred_net = pe.prepare_prediction_net(pred_file.name, CAFFE2_DB_TYPE)
                test_inputs = self._get_rand_input(
                    BATCH_SIZE,
                    W_VOCAB_SIZE,
                    DICT_VOCAB_SIZE,
                    test_num_words,
                    test_num_dict_feat,
                )
                self._feed_c2_input(workspace, test_inputs, metadata.feature_itos_map)
                workspace.RunNetOnce(pred_net)
                c2_out = [list(workspace.FetchBlob(o_name)) for o_name in output_names]

                py_model.eval()
                py_outs = py_model(*test_inputs)
                # Do log_softmax since we do that before exporting predictor nets
                py_outs = F.log_softmax(py_outs, 1)
                np.testing.assert_array_almost_equal(
                    py_outs.view(-1).detach().numpy(), np.array(c2_out).flatten()
                )

    @given(
        export_num_words=st.integers(1, 5),
        export_num_dict_feat=st.integers(1, 6),
        num_word_classes=st.integers(2, 5),
        test_num_words=st.integers(1, 7),
        test_num_dict_feat=st.integers(1, 8),
        num_predictions=st.integers(2, 5),
    )
    def test_wordblstm_export_to_caffe2(
        self,
        export_num_words,
        export_num_dict_feat,
        num_word_classes,
        test_num_words,
        test_num_dict_feat,
        num_predictions,
    ):
        for WORD_CONFIG in WORD_CONFIGS:
            config = self._get_config(WordTaggingJobSpec, WORD_CONFIG)
            metadata = self._get_metadata(0, num_word_classes)
            py_model = create_model(config.model, config.features, metadata)
            exporter = create_exporter(
                config.exporter, config.features, config.labels, metadata
            )
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".{}".format(".predictor")
            ) as pred_file:
                output_names = exporter.export_to_caffe2(py_model, pred_file.name)
                workspace.ResetWorkspace()
            pred_net = pe.prepare_prediction_net(pred_file.name, CAFFE2_DB_TYPE)
            for _i in range(num_predictions):
                test_inputs = self._get_rand_input(
                    BATCH_SIZE,
                    W_VOCAB_SIZE,
                    DICT_VOCAB_SIZE,
                    test_num_words,
                    test_num_dict_feat,
                )
                self._feed_c2_input(workspace, test_inputs, metadata.feature_itos_map)
                workspace.RunNetOnce(pred_net)
                c2_out = [list(workspace.FetchBlob(o_name)) for o_name in output_names]
                py_model.eval()
                py_outs = py_model(*test_inputs)
                context = {SEQ_LENS: test_inputs[1]}
                pred, score = py_model.get_pred(py_outs, context)

                np.testing.assert_array_almost_equal(
                    torch.transpose(score, 1, 2).contiguous().view(-1).detach().numpy(),
                    np.array(c2_out).flatten(),
                )

    @given(
        export_num_words=st.integers(1, 5),
        export_num_dict_feat=st.integers(1, 6),
        num_doc_classes=st.integers(2, 5),
        num_word_classes=st.integers(2, 4),
        test_num_words=st.integers(1, 7),
        test_num_dict_feat=st.integers(1, 8),
        num_predictions=st.integers(1, 5),
    )
    def test_joint_export_to_caffe2(
        self,
        export_num_words,
        export_num_dict_feat,
        num_doc_classes,
        num_word_classes,
        test_num_words,
        test_num_dict_feat,
        num_predictions,
    ):
        config = self._get_config(JointTextJobSpec, JOINT_CONFIG)
        metadata = self._get_metadata(num_doc_classes, num_word_classes)
        py_model = create_model(config.model, config.features, metadata)
        exporter = create_exporter(
            config.exporter, config.features, config.labels, metadata
        )
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".{}".format(".predictor")
        ) as pred_file:
            exporter.export_to_caffe2(py_model, pred_file.name)
            workspace.ResetWorkspace()

        pred_net = pe.prepare_prediction_net(pred_file.name, CAFFE2_DB_TYPE)

        for _i in range(num_predictions):
            test_inputs = self._get_rand_input(
                BATCH_SIZE,
                W_VOCAB_SIZE,
                DICT_VOCAB_SIZE,
                test_num_words,
                test_num_dict_feat,
            )
            self._feed_c2_input(workspace, test_inputs, metadata.feature_itos_map)
            workspace.RunNetOnce(pred_net)
            doc_output_names = [
                "{}:{}".format("doc_scores", class_name)
                for class_name in metadata.label_names[0]
            ]
            word_output_names = [
                "{}:{}".format("word_scores", class_name)
                for class_name in metadata.label_names[1]
            ]

            py_model.eval()
            logits = py_model(*test_inputs)
            context = {SEQ_LENS: test_inputs[1]}
            (d_pred, w_pred), (d_score, w_score) = py_model.get_pred(logits, context)

            c2_doc_out = []
            for o_name in doc_output_names:
                c2_doc_out.extend(list(workspace.FetchBlob(o_name)))
            np.testing.assert_array_almost_equal(
                d_score.view(-1).detach().numpy(), np.array(c2_doc_out).flatten()
            )

            c2_word_out = []
            for o_name in word_output_names:
                c2_word_out.extend(list(workspace.FetchBlob(o_name)))

            np.testing.assert_array_almost_equal(
                torch.transpose(w_score, 1, 2).contiguous().view(-1).detach().numpy(),
                np.array(c2_word_out).flatten(),
            )

    def _get_metadata(self, num_doc_classes, num_word_classes):
        labels = {}
        if num_doc_classes:
            vocab = Vocab(Counter())
            vocab.itos = ["C_{}".format(i) for i in range(num_doc_classes)]
            label_meta = FieldMeta()
            label_meta.vocab_size = num_doc_classes
            label_meta.vocab = vocab
            labels[DatasetFieldName.DOC_LABEL_FIELD] = label_meta

        if num_word_classes:
            vocab = Vocab(Counter())
            vocab.itos = ["W_{}".format(i) for i in range(num_word_classes)]
            label_meta = FieldMeta()
            label_meta.vocab_size = num_word_classes
            label_meta.vocab = vocab
            label_meta.pad_token_idx = 0
            labels[DatasetFieldName.WORD_LABEL_FIELD] = label_meta

        w_vocab = Vocab(Counter())
        dict_vocab = Vocab(Counter())
        w_vocab.itos = W_VOCAB
        dict_vocab.itos = DICT_VOCAB

        text_feat_meta = FieldMeta()
        text_feat_meta.unk_token_idx = UNK_IDX
        text_feat_meta.pad_token_idx = PAD_IDX
        text_feat_meta.vocab_size = W_VOCAB_SIZE
        text_feat_meta.vocab = w_vocab
        text_feat_meta.vocab_export_name = PredictorInputNames.TOKENS_IDS

        dict_feat_meat = FieldMeta()
        dict_feat_meat.vocab_size = DICT_VOCAB_SIZE
        dict_feat_meat.vocab = dict_vocab
        dict_feat_meat.vocab_export_name = PredictorInputNames.DICT_FEAT_IDS

        meta = CommonMetadata()
        meta.features = {
            DatasetFieldName.TEXT_FIELD: text_feat_meta,
            DatasetFieldName.DICT_FIELD: dict_feat_meat,
        }
        meta.labels = labels
        meta.label_names = [label.vocab.itos for label in labels.values()]
        meta.feature_itos_map = {
            f.vocab_export_name: f.vocab.itos for _, f in meta.features.items()
        }
        return meta

    def _get_rand_input(
        self, batch_size, w_vocab_size, d_vocab_size, num_words, num_dict_feats
    ):
        text = torch.from_numpy(
            np.random.randint(w_vocab_size, size=(batch_size, num_words)).astype(
                np.int64
            )
        )
        lengths = torch.from_numpy(
            np.random.randint(num_words, num_words + 1, size=(batch_size)).astype(
                np.int64
            )
        )
        dict_feat = torch.from_numpy(
            np.random.randint(
                d_vocab_size, size=(batch_size, num_dict_feats * num_words)
            ).astype(np.int64)
        )
        dict_weights = torch.from_numpy(
            np.random.randn(batch_size, num_words * num_dict_feats).astype(np.float32)
        )
        dict_lengths = torch.from_numpy(
            np.random.randint(
                1, num_dict_feats + 1, size=(num_words * batch_size)
            ).astype(np.int64)
        )

        return (text, lengths, (dict_feat, dict_weights, dict_lengths))

    def _get_config(self, cls, config_str):
        params_json = json.loads(config_str)
        config = config_from_json(cls, params_json)
        return config

    def _feed_c2_input(self, workspace, py_inputs, vocab_map):
        c2_input = []

        for py_input in py_inputs:
            c2_input = c2_input + (
                list(py_input) if isinstance(py_input, tuple) else [py_input]
            )
        for i, input in enumerate(list(c2_input)):
            input_np = input.numpy()
            if INPUT_NAMES[i] in vocab_map.keys():
                # Map the input to the str form
                input_vocab = vocab_map[INPUT_NAMES[i]]
                input_str = [[input_vocab[x] for x in ex] for ex in input_np]
                input_np = np.array(input_str, dtype=str)
                workspace.FeedBlob(INPUT_NAMES[i] + "_str:value", input_np)
            else:
                workspace.FeedBlob(INPUT_NAMES[i], input_np)

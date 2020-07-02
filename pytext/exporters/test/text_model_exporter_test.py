#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

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
from pytext.builtin_task import (
    DocumentClassificationTask,
    IntentSlotTask,
    SeqNNTask,
    WordTaggingTask,
)
from pytext.common.constants import DatasetFieldName, SpecialTokens
from pytext.config import config_from_json
from pytext.config.component import create_exporter, create_model
from pytext.data import CommonMetadata
from pytext.data.utils import Vocabulary
from pytext.exporters.exporter import ModelExporter
from pytext.fields import (
    CharFeatureField,
    DictFeatureField,
    FieldMeta,
    SeqFeatureField,
    TextFeatureField,
)
from pytext.task.new_task import _NewTask
from pytext.utils.onnx import CAFFE2_DB_TYPE
from torchtext.vocab import Vocab


JOINT_CONFIG = """
{
  "model": {
    "representation": {
      "BiLSTMDocSlotAttention": {
        "lstm": {
          "BiLSTM": {
             "lstm_dim": 30,
             "num_layers": 1
          }
        },
        "pooling": {
          "SelfAttention": {
            "attn_dimension": 30,
            "dropout": 0.3
          }
        }
      }
    },
    "decoder": {
      "use_doc_probs_in_word": true
    },
    "output_layer": {
      "doc_output": {
        "loss": {
          "CrossEntropyLoss": {}
        }
      },
      "word_output": {
        "CRFOutputLayer": {}
      }
    }
  }
}
"""

DOC_CONFIGS = [
    """
{
  "model": {
    "representation": {
      "DocNNRepresentation": {}
    },
    "output_layer": {
      "loss": {
        "CrossEntropyLoss": {}
      }
    }
  },
  "features": {
    "word_feat": {},
    "dict_feat": {},
    "char_feat": {
      "embed_dim": 5,
      "cnn": {
        "kernel_num": 2,
        "kernel_sizes": [2, 3]
        }
      },
      "dense_feat": {
        "dim":10
      }
  },
  "featurizer": {
    "SimpleFeaturizer": {}
  },
  "trainer": {
    "epochs": 1
  },
  "exporter": {}
}
""",
    """
{
  "model": {
    "representation": {
        "BiLSTMDocAttention": {
        "pooling": {
          "MaxPool": {}
        }
      }
    },
    "output_layer": {
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
  "featurizer": {
    "SimpleFeaturizer": {}
  },
  "trainer": {
    "epochs": 1
  },
  "exporter": {}
}
""",
    """
{
  "model": {
    "representation": {
      "DocNNRepresentation": {}
    },
    "output_layer": {
      "loss": {
        "CrossEntropyLoss": {}
      }
    }
  },
  "features": {
    "word_feat": {},
    "dict_feat": {},
    "char_feat": {
      "embed_dim": 5,
      "cnn": {
        "kernel_num": 2,
        "kernel_sizes": [2, 3]
        }
      }
  },
  "featurizer": {
    "SimpleFeaturizer": {}
  },
  "trainer": {
    "epochs": 1
  },
  "exporter": {}
}
""",
]

DOC_CONFIGS_WITH_EXPORT_LOGITS = [
    """
{
  "model": {
    "representation": {
        "BiLSTMDocAttention": {
        "pooling": {
          "MaxPool": {}
        }
      }
    },
    "output_layer": {
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
  "featurizer": {
    "SimpleFeaturizer": {}
  },
  "trainer": {
    "epochs": 1
  },
  "exporter": {
    "export_logits": true
  }
}
"""
]

WORD_CONFIGS = [
    """
{
  "model": {
    "representation": {
      "BiLSTMSlotAttention": {
        "lstm": {
          "lstm_dim": 30,
          "num_layers": 2
        }
      }
    },
    "output_layer": {
      "WordTaggingOutputLayer": {}
    }
  }
}
""",
    """
{
  "model": {
    "representation": {
      "BiLSTMSlotAttention": {
        "lstm": {
          "lstm_dim": 30,
          "num_layers": 2
        }
      }
    },
    "output_layer": {
      "CRFOutputLayer": {}
    }
  }
}
""",
]


SEQ_NN_CONFIG = """
 {
  "model": {
    "representation": {
      "doc_representation": {},
      "seq_representation": {
        "DocNNRepresentation": {}
      }
    }
  }
}
"""


CONTEXTUAL_INTENT_SLOT_CONFIG = """
{
    "trainer": {
      "epochs": 1
    },
    "metric_reporter": {
      "IntentSlotMetricReporter": {}
    },
    "model": {
      "ContextualIntentSlotModel": {
        "inputs": {
          "tokens": {
          },
          "seq_tokens": {}
        },
        "word_embedding": {
          "embed_dim": 10
        },
        "seq_embedding": {
          "embed_dim": 10
        }
      }
    }
}
"""
WORD_VOCAB = [SpecialTokens.UNK, "W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8", "W9"]


W_VOCAB_SIZE = 10
UNK_IDX = 0
PAD_IDX = 1
W_VOCAB = ["<UNK>", "W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8", "W9"]
DICT_VOCAB_SIZE = 10
DICT_VOCAB = ["<UNK>", "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9"]
CHAR_VOCAB_SIZE = 10
CHAR_VOCAB = ["<UNK>", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]

# For now we need to fix the batch_size for exporting and testing,
# Need to remove this and make it a random input once ONNX is able to
# Handle different batch_sizes
BATCH_SIZE = 1

# Fixed dimension of dense_features since it needs to be specified in config
DENSE_FEATURE_DIM = 10


class ModelExporterTest(hu.HypothesisTestCase):
    @given(
        export_num_words=st.integers(1, 5),
        export_num_dict_feat=st.integers(1, 6),
        num_doc_classes=st.integers(2, 5),
        test_num_words=st.integers(1, 7),
        test_num_dict_feat=st.integers(1, 8),
        num_predictions=st.integers(1, 4),
        test_num_chars=st.integers(1, 7),
    )
    # TODO () Port this test to DocumentClassificationTask
    def DISABLED_test_doc_export_to_caffe2(
        self,
        export_num_words,
        export_num_dict_feat,
        num_doc_classes,
        test_num_words,
        test_num_dict_feat,
        num_predictions,
        test_num_chars,
    ):
        for config in DOC_CONFIGS:
            config = self._get_config(DocumentClassificationTask.Config, config)
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
                    config.features,
                    BATCH_SIZE,
                    W_VOCAB_SIZE,
                    DICT_VOCAB_SIZE,
                    CHAR_VOCAB_SIZE,
                    test_num_words,
                    test_num_dict_feat,
                    test_num_chars,
                )
                self._feed_c2_input(
                    workspace,
                    test_inputs,
                    exporter.input_names,
                    metadata.feature_itos_map,
                )
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
        num_doc_classes=st.integers(2, 5),
        test_num_words=st.integers(1, 7),
        test_num_dict_feat=st.integers(1, 8),
        num_predictions=st.integers(1, 4),
        test_num_chars=st.integers(1, 7),
    )
    # TODO () Port this test to DocumentClassificationTask
    def DISABLED_test_doc_export_to_caffe2_with_logits(
        self,
        num_doc_classes,
        test_num_words,
        test_num_dict_feat,
        num_predictions,
        test_num_chars,
    ):
        for config in DOC_CONFIGS_WITH_EXPORT_LOGITS:
            config = self._get_config(DocumentClassificationTask.Config, config)
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
                    config.features,
                    BATCH_SIZE,
                    W_VOCAB_SIZE,
                    DICT_VOCAB_SIZE,
                    CHAR_VOCAB_SIZE,
                    test_num_words,
                    test_num_dict_feat,
                    test_num_chars,
                )
                self._feed_c2_input(
                    workspace,
                    test_inputs,
                    exporter.input_names,
                    metadata.feature_itos_map,
                )
                workspace.RunNetOnce(pred_net)
                c2_out = [list(workspace.FetchBlob(o_name)) for o_name in output_names]

                py_model.eval()
                py_outs = py_model(*test_inputs)
                np.testing.assert_array_almost_equal(
                    py_outs.view(-1).detach().numpy(), np.array(c2_out[-1]).flatten()
                )

                # Do log_softmax since we do that before exporting predictor nets
                py_outs = F.log_softmax(py_outs, 1)
                np.testing.assert_array_almost_equal(
                    py_outs.view(-1).detach().numpy(), np.array(c2_out[:-1]).flatten()
                )

    @given(
        export_num_words=st.integers(1, 5),
        num_word_classes=st.integers(2, 5),
        test_num_words=st.integers(1, 7),
        num_predictions=st.integers(2, 5),
    )
    def test_wordblstm_export_to_caffe2(
        self, export_num_words, num_word_classes, test_num_words, num_predictions
    ):
        for WORD_CONFIG in WORD_CONFIGS:
            config = self._get_config(WordTaggingTask.Config, WORD_CONFIG)
            tensorizers, data = _NewTask._init_tensorizers(config)
            word_labels = [SpecialTokens.PAD, SpecialTokens.UNK, "NoLabel", "person"]
            tensorizers["labels"].vocab = Vocabulary(word_labels)
            tensorizers["tokens"].vocab = Vocabulary(WORD_VOCAB)
            py_model = _NewTask._init_model(config.model, tensorizers)
            dummy_test_input = self._get_rand_input_intent_slot(
                BATCH_SIZE, W_VOCAB_SIZE, test_num_words
            )
            exporter = ModelExporter(
                ModelExporter.Config(),
                py_model.get_export_input_names(tensorizers),
                dummy_test_input,
                py_model.vocab_to_export(tensorizers),
                py_model.get_export_output_names(tensorizers),
            )
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".{}".format(".predictor")
            ) as pred_file:
                exporter.export_to_caffe2(py_model, pred_file.name)
                workspace.ResetWorkspace()
            pred_net = pe.prepare_prediction_net(pred_file.name, CAFFE2_DB_TYPE)
            for _i in range(num_predictions):
                test_inputs = self._get_rand_input_intent_slot(
                    BATCH_SIZE, W_VOCAB_SIZE, test_num_words
                )
                self._feed_c2_input(
                    workspace, test_inputs, exporter.input_names, exporter.vocab_map
                )
                workspace.RunNetOnce(pred_net)
                word_output_names = [
                    "{}:{}".format("word_scores", class_name)
                    for class_name in word_labels
                ]
                py_model.eval()
                py_outs = py_model(*test_inputs)
                context = {"seq_lens": test_inputs[-1]}
                target = None
                pred, score = py_model.get_pred(py_outs, target, context)
                c2_word_out = []
                for o_name in word_output_names:
                    c2_word_out.extend(list(workspace.FetchBlob(o_name)))

                np.testing.assert_array_almost_equal(
                    torch.transpose(score, 1, 2).contiguous().view(-1).detach().numpy(),
                    np.array(c2_word_out).flatten(),
                )

    def _get_rand_input_intent_slot(
        self, batch_size, w_vocab_size, num_words, num_seq=0
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
        inputs = [text]
        if num_seq > 0:
            inputs.append(
                torch.from_numpy(
                    np.random.randint(
                        w_vocab_size, size=(batch_size, num_seq, num_words)
                    ).astype(np.int64)
                )
            )
        inputs.append(lengths)
        if num_seq > 0:
            inputs.append(
                torch.from_numpy(
                    np.random.randint(num_seq, num_seq + 1, size=(batch_size)).astype(
                        np.int64
                    )
                )
            )
        return tuple(inputs)

    @given(
        export_num_words=st.integers(1, 5),
        num_doc_classes=st.integers(2, 5),
        num_word_classes=st.integers(2, 4),
        test_num_words=st.integers(1, 7),
        num_predictions=st.integers(1, 5),
    )
    def test_joint_export_to_caffe2(
        self,
        export_num_words,
        num_doc_classes,
        num_word_classes,
        test_num_words,
        num_predictions,
    ):
        config = self._get_config(IntentSlotTask.Config, JOINT_CONFIG)
        tensorizers, data = _NewTask._init_tensorizers(config)
        doc_labels = [SpecialTokens.UNK, "cu:other", "cu:address_Person"]
        word_labels = [SpecialTokens.PAD, SpecialTokens.UNK, "NoLabel", "person"]
        tensorizers["word_labels"].vocab = Vocabulary(word_labels)
        tensorizers["doc_labels"].vocab = Vocabulary(doc_labels)
        tensorizers["tokens"].vocab = Vocabulary(WORD_VOCAB)
        py_model = _NewTask._init_model(config.model, tensorizers)
        dummy_test_input = self._get_rand_input_intent_slot(
            BATCH_SIZE, W_VOCAB_SIZE, test_num_words
        )
        exporter = ModelExporter(
            ModelExporter.Config(),
            py_model.get_export_input_names(tensorizers),
            dummy_test_input,
            py_model.vocab_to_export(tensorizers),
            py_model.get_export_output_names(tensorizers),
        )

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".{}".format(".predictor")
        ) as pred_file:
            exporter.export_to_caffe2(py_model, pred_file.name)
            workspace.ResetWorkspace()

        pred_net = pe.prepare_prediction_net(pred_file.name, CAFFE2_DB_TYPE)

        for _i in range(num_predictions):
            test_inputs = self._get_rand_input_intent_slot(
                BATCH_SIZE, W_VOCAB_SIZE, test_num_words
            )
            self._feed_c2_input(
                workspace, test_inputs, exporter.input_names, exporter.vocab_map
            )
            workspace.RunNetOnce(pred_net)
            doc_output_names = [
                "{}:{}".format("doc_scores", class_name) for class_name in doc_labels
            ]
            word_output_names = [
                "{}:{}".format("word_scores", class_name) for class_name in word_labels
            ]

            py_model.eval()
            logits = py_model(*test_inputs)
            context = {"seq_lens": test_inputs[-1]}
            target = None
            (d_pred, w_pred), (d_score, w_score) = py_model.get_pred(
                logits, target, context
            )

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

    @given(
        export_num_words=st.integers(1, 5),
        num_doc_classes=st.integers(2, 5),
        test_num_words=st.integers(1, 7),
        num_predictions=st.integers(1, 5),
        test_num_seq=st.integers(1, 7),
    )
    def test_seq_nn_export_to_caffe2(
        self,
        export_num_words,
        num_doc_classes,
        test_num_words,
        num_predictions,
        test_num_seq,
    ):
        config = self._get_config(SeqNNTask.Config, SEQ_NN_CONFIG)
        tensorizers, data = _NewTask._init_tensorizers(config)
        doc_labels = [SpecialTokens.UNK, "cu:other", "cu:address_Person"]
        tensorizers["labels"].vocab = Vocabulary(doc_labels)
        tensorizers["tokens"].vocab = Vocabulary(WORD_VOCAB)
        py_model = _NewTask._init_model(config.model, tensorizers)
        dummy_test_input = self._get_seq_nn_rand_input(
            BATCH_SIZE, W_VOCAB_SIZE, test_num_words, test_num_seq
        )
        exporter = ModelExporter(
            ModelExporter.Config(),
            py_model.get_export_input_names(tensorizers),
            dummy_test_input,
            py_model.vocab_to_export(tensorizers),
            py_model.get_export_output_names(tensorizers),
        )
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".{}".format(".predictor")
        ) as pred_file:
            output_names = exporter.export_to_caffe2(py_model, pred_file.name)
            workspace.ResetWorkspace()

        pred_net = pe.prepare_prediction_net(pred_file.name, CAFFE2_DB_TYPE)
        for _i in range(num_predictions):
            test_inputs = self._get_seq_nn_rand_input(
                BATCH_SIZE, W_VOCAB_SIZE, test_num_words, test_num_seq
            )
            self._feed_c2_input(
                workspace, test_inputs, exporter.input_names, exporter.vocab_map
            )
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
        test_num_words=st.integers(1, 7),
        num_predictions=st.integers(1, 5),
        test_num_seq=st.integers(1, 7),
    )
    def test_contextual_intent_slot_export_to_caffe2(
        self, test_num_words, num_predictions, test_num_seq
    ):
        config = self._get_config(IntentSlotTask.Config, CONTEXTUAL_INTENT_SLOT_CONFIG)
        tensorizers, data = _NewTask._init_tensorizers(config)
        doc_labels = [SpecialTokens.UNK, "cu:other", "cu:address_Person"]
        word_labels = [SpecialTokens.UNK, "NoLabel", "person"]
        tensorizers["word_labels"].vocab = Vocabulary(word_labels)
        tensorizers["doc_labels"].vocab = Vocabulary(doc_labels)
        tensorizers["tokens"].vocab = Vocabulary(WORD_VOCAB)
        tensorizers["seq_tokens"].vocab = Vocabulary(WORD_VOCAB)
        py_model = _NewTask._init_model(config.model, tensorizers)
        dummy_test_input = self._get_rand_input_intent_slot(
            BATCH_SIZE, W_VOCAB_SIZE, test_num_words, test_num_seq
        )
        exporter = ModelExporter(
            ModelExporter.Config(),
            py_model.get_export_input_names(tensorizers),
            dummy_test_input,
            py_model.vocab_to_export(tensorizers),
            py_model.get_export_output_names(tensorizers),
        )

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".{}".format(".predictor")
        ) as pred_file:
            print(pred_file.name)
            exporter.export_to_caffe2(py_model, pred_file.name)
            workspace.ResetWorkspace()

        pred_net = pe.prepare_prediction_net(pred_file.name, CAFFE2_DB_TYPE)
        for _i in range(num_predictions):
            test_inputs = self._get_rand_input_intent_slot(
                BATCH_SIZE, W_VOCAB_SIZE, test_num_words, test_num_seq
            )
            self._feed_c2_input(
                workspace, test_inputs, exporter.input_names, exporter.vocab_map
            )
            workspace.RunNetOnce(pred_net)
            doc_output_names = [
                "{}:{}".format("doc_scores", class_name) for class_name in doc_labels
            ]
            word_output_names = [
                "{}:{}".format("word_scores", class_name) for class_name in word_labels
            ]
            py_model.eval()
            logits = py_model(*test_inputs)
            context = {"seq_lens": test_inputs[-1]}
            target = None
            (d_pred, w_pred), (d_score, w_score) = py_model.get_pred(
                logits, target, context
            )

            c2_doc_out = []
            for o_name in doc_output_names:
                c2_doc_out.extend(list(workspace.FetchBlob(o_name)))
            c2_word_out = []
            for o_name in word_output_names:
                c2_word_out.extend(list(workspace.FetchBlob(o_name)))

            np.testing.assert_array_almost_equal(
                d_score.view(-1).detach().numpy(), np.array(c2_doc_out).flatten()
            )

            np.testing.assert_array_almost_equal(
                torch.transpose(w_score, 1, 2).contiguous().view(-1).detach().numpy(),
                np.array(c2_word_out).flatten(),
            )

    def _get_metadata(self, num_doc_classes, num_word_classes):
        labels = []
        if num_doc_classes:
            vocab = Vocab(Counter())
            vocab.itos = ["C_{}".format(i) for i in range(num_doc_classes)]
            label_meta = FieldMeta()
            label_meta.vocab_size = num_doc_classes
            label_meta.vocab = vocab
            labels.append(label_meta)

        if num_word_classes:
            vocab = Vocab(Counter())
            vocab.itos = ["W_{}".format(i) for i in range(num_word_classes)]
            label_meta = FieldMeta()
            label_meta.vocab_size = num_word_classes
            label_meta.vocab = vocab
            label_meta.pad_token_idx = 0
            labels.append(label_meta)

        w_vocab = Vocab(Counter())
        dict_vocab = Vocab(Counter())
        c_vocab = Vocab(Counter())
        d_vocab = Vocab(Counter())
        w_vocab.itos = W_VOCAB
        dict_vocab.itos = DICT_VOCAB
        c_vocab.itos = CHAR_VOCAB
        d_vocab.itos = []

        text_feat_meta = FieldMeta()
        text_feat_meta.unk_token_idx = UNK_IDX
        text_feat_meta.pad_token_idx = PAD_IDX
        text_feat_meta.vocab_size = W_VOCAB_SIZE
        text_feat_meta.vocab = w_vocab
        text_feat_meta.vocab_export_name = "tokens_vals"
        text_feat_meta.pretrained_embeds_weight = None
        text_feat_meta.dummy_model_input = TextFeatureField.dummy_model_input

        dict_feat_meta = FieldMeta()
        dict_feat_meta.vocab_size = DICT_VOCAB_SIZE
        dict_feat_meta.vocab = dict_vocab
        dict_feat_meta.vocab_export_name = "dict_vals"
        dict_feat_meta.pretrained_embeds_weight = None
        dict_feat_meta.dummy_model_input = DictFeatureField.dummy_model_input

        char_feat_meta = FieldMeta()
        char_feat_meta.vocab_size = CHAR_VOCAB_SIZE
        char_feat_meta.vocab = c_vocab
        char_feat_meta.vocab_export_name = "char_vals"
        char_feat_meta.pretrained_embeds_weight = None
        char_feat_meta.dummy_model_input = CharFeatureField.dummy_model_input

        dense_feat_meta = FieldMeta()
        dense_feat_meta.vocab_size = 0
        dense_feat_meta.vocab = d_vocab
        dense_feat_meta.vocab_export_name = "dense_vals"
        dense_feat_meta.pretrained_embeds_weight = None
        # ugh, dims are fixed
        dense_feat_meta.dummy_model_input = torch.tensor(
            [[1.0] * DENSE_FEATURE_DIM, [1.0] * DENSE_FEATURE_DIM],
            dtype=torch.float,
            device="cpu",
        )

        seq_feat_meta = FieldMeta()
        seq_feat_meta.unk_token_idx = UNK_IDX
        seq_feat_meta.pad_token_idx = PAD_IDX
        seq_feat_meta.vocab_size = W_VOCAB_SIZE
        seq_feat_meta.vocab = w_vocab
        seq_feat_meta.vocab_export_name = "seq_tokens_vals"
        seq_feat_meta.pretrained_embeds_weight = None
        seq_feat_meta.dummy_model_input = SeqFeatureField.dummy_model_input

        meta = CommonMetadata()
        meta.features = {
            DatasetFieldName.TEXT_FIELD: text_feat_meta,
            DatasetFieldName.DICT_FIELD: dict_feat_meta,
            DatasetFieldName.CHAR_FIELD: char_feat_meta,
            DatasetFieldName.DENSE_FIELD: dense_feat_meta,
            DatasetFieldName.SEQ_FIELD: seq_feat_meta,
        }
        meta.target = labels
        if len(labels) == 1:
            [meta.target] = meta.target
        meta.label_names = [label.vocab.itos for label in labels]
        meta.feature_itos_map = {
            f.vocab_export_name: f.vocab.itos for _, f in meta.features.items()
        }
        return meta

    def _get_seq_nn_rand_input(self, batch_size, w_vocab_size, num_words, num_seq=1):
        seq = torch.from_numpy(
            np.random.randint(
                w_vocab_size, size=(batch_size, num_seq, num_words)
            ).astype(np.int64)
        )
        seq_lengths = torch.from_numpy(
            np.random.randint(num_seq, num_seq + 1, size=(batch_size)).astype(np.int64)
        )
        return (seq, seq_lengths)

    def _get_rand_input(
        self,
        features,
        batch_size,
        w_vocab_size,
        d_vocab_size,
        c_vocab_size,
        num_words,
        num_dict_feats,
        num_chars,
        num_seq=1,
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
                1, num_dict_feats + 1, size=(batch_size, num_words)
            ).astype(np.int64)
        )
        chars = torch.from_numpy(
            np.random.randint(
                c_vocab_size, size=(batch_size, num_words, num_chars)
            ).astype(np.int64)
        )
        dense_features = torch.from_numpy(
            np.random.rand(batch_size, DENSE_FEATURE_DIM).astype(np.float32)
        )
        inputs = []
        if features.word_feat:
            inputs.append(text)
        if features.dict_feat:
            inputs.append((dict_feat, dict_weights, dict_lengths))
        if features.char_feat:
            inputs.append(chars)
        if getattr(features, "seq_word_feat", False):
            inputs.append(
                torch.from_numpy(
                    np.random.randint(
                        w_vocab_size, size=(batch_size, num_seq, num_words)
                    ).astype(np.int64)
                )
            )
        inputs.append(lengths)
        if getattr(features, "seq_word_feat", False):
            inputs.append(
                torch.from_numpy(
                    np.random.randint(num_seq, num_seq + 1, size=(batch_size)).astype(
                        np.int64
                    )
                )
            )
        if features.dense_feat:
            inputs.append(dense_features)
        return tuple(inputs)

    def _get_config(self, cls, config_str):
        params_json = json.loads(config_str)
        config = config_from_json(cls, params_json)
        return config

    def _feed_c2_input(self, workspace, py_inputs, input_names, vocab_map):
        c2_input = []

        for py_input in py_inputs:
            c2_input = c2_input + (
                list(py_input) if isinstance(py_input, tuple) else [py_input]
            )
        for i, input in enumerate(list(c2_input)):
            input_np = input.numpy()
            if input_names[i] in vocab_map.keys():
                # Map the input to the str form
                input_vocab = vocab_map[input_names[i]]
                map_fn = np.vectorize(lambda x: input_vocab[x])
                input_str = map_fn(input_np)
                input_np = np.array(input_str, dtype=str)
                workspace.FeedBlob(input_names[i] + "_str:value", input_np)
            else:
                workspace.FeedBlob(input_names[i], input_np)

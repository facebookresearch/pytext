#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reservedimport unittest
import unittest

import torch
from pytext.common.constants import Stage
from pytext.data import Batcher
from pytext.data.data import Data
from pytext.data.sources.data_source import Gazetteer
from pytext.data.sources.tsv import TSVDataSource
from pytext.data.tensorizers import (
    ByteTokenTensorizer,
    GazetteerTensorizer,
    TokenTensorizer,
    initialize_tensorizers,
)
from pytext.models.embeddings import (
    ContextualTokenEmbedding,
    DictEmbedding,
    WordEmbedding,
)
from pytext.models.seq_models.rnn_encoder import LSTMSequenceEncoder
from pytext.models.seq_models.rnn_encoder_decoder import RNNModel
from pytext.models.seq_models.seq2seq_model import Seq2SeqModel

# @dep //pytext/utils:utils_lib
from pytext.utils.test import import_tests_module


tests_module = import_tests_module()


TEST_FILE_NAME = tests_module.test_file("seq2seq_model_unit.tsv")


def get_tensorizers(add_dict_feat=False, add_contextual_feat=False):
    schema = {"source_sequence": str, "dict_feat": Gazetteer, "target_sequence": str}
    data_source = TSVDataSource.from_config(
        TSVDataSource.Config(
            train_filename=TEST_FILE_NAME,
            field_names=["source_sequence", "dict_feat", "target_sequence"],
        ),
        schema,
    )
    src_tensorizer = TokenTensorizer.from_config(
        TokenTensorizer.Config(
            column="source_sequence", add_eos_token=True, add_bos_token=True
        )
    )
    tgt_tensorizer = TokenTensorizer.from_config(
        TokenTensorizer.Config(
            column="target_sequence", add_eos_token=True, add_bos_token=True
        )
    )
    tensorizers = {"src_seq_tokens": src_tensorizer, "trg_seq_tokens": tgt_tensorizer}
    initialize_tensorizers(tensorizers, data_source.train)

    if add_dict_feat:
        tensorizers["dict_feat"] = GazetteerTensorizer.from_config(
            GazetteerTensorizer.Config(
                text_column="source_sequence", dict_column="dict_feat"
            )
        )
        initialize_tensorizers(
            {"dict_feat": tensorizers["dict_feat"]}, data_source.train
        )

    if add_contextual_feat:
        tensorizers["contextual_token_embedding"] = ByteTokenTensorizer.from_config(
            ByteTokenTensorizer.Config(column="source_sequence")
        )
        initialize_tensorizers(
            {"contextual_token_embedding": tensorizers["contextual_token_embedding"]},
            data_source.train,
        )

    return tensorizers


# Smoke tests that call torchscriptify and execute the model for all the cases.
# This should at least make sure we're testing end to end.
class Seq2SeqModelExportTests(unittest.TestCase):
    def test_tokens(self):
        # TODO (T65593688): this should be removed after
        # https://github.com/pytorch/pytorch/pull/33645 is merged.
        with torch.no_grad():
            model = Seq2SeqModel.from_config(
                Seq2SeqModel.Config(
                    source_embedding=WordEmbedding.Config(embed_dim=512),
                    target_embedding=WordEmbedding.Config(embed_dim=512),
                ),
                get_tensorizers(),
            )
            model.eval()
            ts_model = model.torchscriptify()
            res = ts_model(["call", "mom"])
            assert res is not None

    def test_tokens_contextual(self):
        # TODO (T65593688): this should be removed after
        # https://github.com/pytorch/pytorch/pull/33645 is merged.
        with torch.no_grad():
            model = Seq2SeqModel.from_config(
                Seq2SeqModel.Config(
                    source_embedding=WordEmbedding.Config(embed_dim=512),
                    target_embedding=WordEmbedding.Config(embed_dim=512),
                    inputs=Seq2SeqModel.Config.ModelInput(
                        contextual_token_embedding=ByteTokenTensorizer.Config()
                    ),
                    contextual_token_embedding=ContextualTokenEmbedding.Config(
                        embed_dim=7
                    ),
                    encoder_decoder=RNNModel.Config(
                        encoder=LSTMSequenceEncoder.Config(embed_dim=519)
                    ),
                ),
                get_tensorizers(add_contextual_feat=True),
            )
            model.eval()
            ts_model = model.torchscriptify()
            res = ts_model(["call", "mom"], contextual_token_embedding=[0.42] * (7 * 2))
            assert res is not None

    def test_tokens_dictfeat(self):
        # TODO (T65593688): this should be removed after
        # https://github.com/pytorch/pytorch/pull/33645 is merged.
        with torch.no_grad():
            model = Seq2SeqModel.from_config(
                Seq2SeqModel.Config(
                    source_embedding=WordEmbedding.Config(embed_dim=512),
                    target_embedding=WordEmbedding.Config(embed_dim=512),
                    inputs=Seq2SeqModel.Config.ModelInput(
                        dict_feat=GazetteerTensorizer.Config(
                            text_column="source_sequence"
                        )
                    ),
                    encoder_decoder=RNNModel.Config(
                        encoder=LSTMSequenceEncoder.Config(embed_dim=612)
                    ),
                    dict_embedding=DictEmbedding.Config(),
                ),
                get_tensorizers(add_dict_feat=True),
            )
            model.eval()
            ts_model = model.torchscriptify()
            res = ts_model(["call", "mom"], (["call", "mom"], [0.42, 0.17], [4, 3]))
            assert res is not None

    def test_tokens_dictfeat_contextual(self):
        # TODO (T65593688): this should be removed after
        # https://github.com/pytorch/pytorch/pull/33645 is merged.
        with torch.no_grad():
            model = Seq2SeqModel.from_config(
                Seq2SeqModel.Config(
                    source_embedding=WordEmbedding.Config(embed_dim=512),
                    target_embedding=WordEmbedding.Config(embed_dim=512),
                    inputs=Seq2SeqModel.Config.ModelInput(
                        dict_feat=GazetteerTensorizer.Config(
                            text_column="source_sequence"
                        ),
                        contextual_token_embedding=ByteTokenTensorizer.Config(),
                    ),
                    encoder_decoder=RNNModel.Config(
                        encoder=LSTMSequenceEncoder.Config(embed_dim=619)
                    ),
                    dict_embedding=DictEmbedding.Config(),
                    contextual_token_embedding=ContextualTokenEmbedding.Config(
                        embed_dim=7
                    ),
                ),
                get_tensorizers(add_dict_feat=True, add_contextual_feat=True),
            )
            model.eval()
            ts_model = model.torchscriptify()
            res = ts_model(
                ["call", "mom"],
                (["call", "mom"], [0.42, 0.17], [4, 3]),
                [0.42] * (7 * 2),
            )
            assert res is not None


# Seq2SeqModel has restrictions on what can happen during evaluation, since
# sequence generation has the opportunity to affect the underlying model.
class Seq2SeqModelEvalTests(unittest.TestCase):
    def test_force_predictions_on_eval(self):
        tensorizers = get_tensorizers()

        model = Seq2SeqModel.from_config(
            Seq2SeqModel.Config(
                source_embedding=WordEmbedding.Config(embed_dim=512),
                target_embedding=WordEmbedding.Config(embed_dim=512),
            ),
            tensorizers,
        )

        # Get sample inputs using a data source.
        schema = {
            "source_sequence": str,
            "dict_feat": Gazetteer,
            "target_sequence": str,
        }
        data = Data.from_config(
            Data.Config(
                source=TSVDataSource.Config(
                    train_filename=TEST_FILE_NAME,
                    field_names=["source_sequence", "dict_feat", "target_sequence"],
                )
            ),
            schema,
            tensorizers,
        )
        data.batcher = Batcher(1, 1, 1)
        raw_batch, batch = next(iter(data.batches(Stage.TRAIN, load_early=True)))
        inputs = model.arrange_model_inputs(batch)

        # Verify that model does not run sequence generation on prediction.
        outputs = model(*inputs)
        pred = model.get_pred(outputs, {"stage": Stage.EVAL})
        self.assertEqual(pred, (None, None))

        # Verify that attempting to set force_eval_predictions is correctly
        # accounted for.
        model.force_eval_predictions = True
        with self.assertRaises(AssertionError):
            _ = model.get_pred(outputs, {"stage": Stage.EVAL})

    def test_reset_incremental_states(self):
        """
        This test might seem trivial. However, interacting with the scripted
        sequence generator crosses the Torchscript boundary, which can lead
        to weird behavior. If the incremental states don't get properly
        reset, the model will produce garbage _after_ the first call, which
        is a pain to debug when you only catch it after training.
        """
        tensorizers = get_tensorizers()

        # Avoid numeric issues with quantization by setting a known seed.
        torch.manual_seed(42)

        model = Seq2SeqModel.from_config(
            Seq2SeqModel.Config(
                source_embedding=WordEmbedding.Config(embed_dim=512),
                target_embedding=WordEmbedding.Config(embed_dim=512),
            ),
            tensorizers,
        )

        # Get sample inputs using a data source.
        schema = {
            "source_sequence": str,
            "dict_feat": Gazetteer,
            "target_sequence": str,
        }
        data = Data.from_config(
            Data.Config(
                source=TSVDataSource.Config(
                    train_filename=TEST_FILE_NAME,
                    field_names=["source_sequence", "dict_feat", "target_sequence"],
                )
            ),
            schema,
            tensorizers,
        )
        data.batcher = Batcher(1, 1, 1)
        raw_batch, batch = next(iter(data.batches(Stage.TRAIN, load_early=True)))
        inputs = model.arrange_model_inputs(batch)

        model.eval()
        outputs = model(*inputs)
        pred, scores = model.get_pred(outputs, {"stage": Stage.TEST})

        # Verify that the incremental states reset correctly.
        decoder = model.sequence_generator.beam_search.decoder_ens
        decoder.reset_incremental_states()
        self.assertDictEqual(decoder.incremental_states, {"0": {}})

        # Verify that the model returns the same predictions.
        new_pred, new_scores = model.get_pred(outputs, {"stage": Stage.TEST})
        self.assertEqual(new_scores, scores)

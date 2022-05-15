#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import unittest

import numpy as np
import torch
from pytext.data.sources.tsv import TSVDataSource
from pytext.data.tensorizers import initialize_tensorizers, TokenTensorizer
from pytext.models.embeddings.word_embedding import WordEmbedding
from pytext.models.seq_models.seq2seq_model import Seq2SeqModel
from pytext.torchscript.seq2seq.scripted_seq2seq_generator import (
    ScriptedSequenceGenerator,
)
from pytext.utils.test import import_tests_module


tests_module = import_tests_module()


TEST_FILE_NAME = tests_module.test_file("compositional_seq2seq_unit.tsv")


def get_single_inference_example(sample_length=10):
    src_tokens = torch.LongTensor(np.ones((sample_length, 1), dtype="int64"))
    src_lengths = torch.IntTensor(np.array([sample_length], dtype="int32"))
    return src_tokens, src_lengths


def get_example_and_check():
    length = np.random.randint(3, 20)
    return (
        get_single_inference_example(length),
        get_single_inference_example(length + 10),
    )


class ScriptedSeq2SeqGeneratorTest(unittest.TestCase):
    def _get_tensorizers(self):
        schema = {"source_sequence": str, "target_sequence": str}
        data_source = TSVDataSource.from_config(
            TSVDataSource.Config(
                train_filename=tests_module.test_file("compositional_seq2seq_unit.tsv"),
                field_names=["source_sequence", "target_sequence"],
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
        tensorizers = {
            "src_seq_tokens": src_tensorizer,
            "trg_seq_tokens": tgt_tensorizer,
        }
        initialize_tensorizers(tensorizers, data_source.train)
        return tensorizers

    def test_generator(self):
        model = Seq2SeqModel.from_config(
            Seq2SeqModel.Config(
                source_embedding=WordEmbedding.Config(embed_dim=512),
                target_embedding=WordEmbedding.Config(embed_dim=512),
            ),
            self._get_tensorizers(),
        )
        sample, _ = get_example_and_check()
        repacked_inputs = {"src_tokens": sample[0].t(), "src_lengths": sample[1]}

        scripted_generator = ScriptedSequenceGenerator(
            [model.model], model.trg_eos_index, ScriptedSequenceGenerator.Config()
        )
        scripted_preds = scripted_generator.generate_hypo(repacked_inputs)

        self.assertIsNotNone(scripted_preds)

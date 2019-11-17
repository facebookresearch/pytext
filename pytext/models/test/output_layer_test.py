#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import random
import string
from typing import Dict, List

import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np
import torch
from hypothesis import given
from pytext.data.tensorizers import LabelTensorizer
from pytext.data.utils import Vocabulary
from pytext.models.output_layers.word_tagging_output_layer import (
    CRFOutputLayer,
    WordTaggingOutputLayer,
)


class OutputLayerTest(hu.HypothesisTestCase):
    def test_create_word_tagging_output_layer(self):
        tensorizer = LabelTensorizer()
        tensorizer.vocab = Vocabulary(["foo", "bar"])
        tensorizer.pad_idx = 0
        layer = WordTaggingOutputLayer.from_config(
            config=WordTaggingOutputLayer.Config(label_weights={"foo": 2.2}),
            labels=tensorizer.vocab,
        )
        np.testing.assert_array_almost_equal(
            np.array([2.2, 1]), layer.loss_fn.weight.detach().numpy()
        )

    @given(
        num_labels=st.integers(2, 6),
        seq_lens=st.lists(
            elements=st.integers(min_value=1, max_value=10), min_size=1, max_size=10
        ),
    )
    def test_torchscript_word_tagging_output_layer(self, num_labels, seq_lens):
        batch_size = len(seq_lens)
        tensorizer = LabelTensorizer()
        vocab_toks: List[str] = [
            OutputLayerTest._generate_random_string() for _ in range(num_labels)
        ]
        tensorizer.vocab = Vocabulary(vocab_toks)
        tensorizer.pad_idx = 0

        word_layer = WordTaggingOutputLayer.from_config(
            config=WordTaggingOutputLayer.Config(), labels=tensorizer.vocab
        )
        crf_layer = CRFOutputLayer.from_config(
            config=CRFOutputLayer.Config(), labels=tensorizer.vocab
        )

        logits, seq_lens_tensor = OutputLayerTest._generate_word_tagging_inputs(
            batch_size, num_labels, seq_lens
        )
        context = {"seq_lens": seq_lens_tensor}

        torchsript_word_layer = word_layer.torchscript_predictions()
        torchscript_crf_layer = crf_layer.torchscript_predictions()

        self._validate_word_tagging_result(
            word_layer.get_pred(logits, None, context)[1],
            torchsript_word_layer(logits, seq_lens_tensor),
            tensorizer.vocab,
        )
        self._validate_word_tagging_result(
            crf_layer.get_pred(logits, None, context)[1],
            torchscript_crf_layer(logits, seq_lens_tensor),
            tensorizer.vocab,
        )

    @staticmethod
    def _generate_random_string(min_size: int = 2, max_size: int = 8) -> str:
        size = random.randint(min_size, max_size)
        return "".join([random.choice(string.ascii_lowercase) for _ in range(size)])

    @staticmethod
    def _generate_word_tagging_inputs(bsize: int, num_labels: int, seq_lens: List[int]):
        max_seq_length = max(seq_lens)
        logits = torch.randn(bsize, max_seq_length, num_labels)
        return logits, torch.tensor(seq_lens, dtype=torch.int)

    def _validate_word_tagging_result(
        self,
        scores: torch.Tensor,
        ts_results: List[List[Dict[str, float]]],
        vocab: Vocabulary,
    ):
        bsize, max_seq_len, num_labels = scores.size()
        self.assertEqual(
            len(ts_results),
            bsize,
            "Batch size must match for pytorch and torchscript class",
        )
        self.assertEqual(
            len(ts_results[0]),
            max_seq_len,
            "Max seq length must match for pytorch and torchscript class",
        )
        self.assertEqual(
            len(ts_results[0][0]),
            num_labels,
            "Number of labels must match for pytorch and torchscript class",
        )
        self.assertEqual(
            len(ts_results[0][0]),
            len(vocab),
            "Number of labels should be same as vocab length",
        )

        for i in range(bsize):
            for j in range(max_seq_len):
                for label, score in ts_results[i][j].items():
                    self.assertAlmostEqual(
                        score,
                        scores[i][j][vocab.idx[label]],
                        (
                            "Scores for [{}][{}][{}] element must match for "
                            "pytorch and torchscript class"
                        ).format(i, j, vocab.idx[label]),
                    )

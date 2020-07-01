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
from pytext.common.constants import SpecialTokens
from pytext.data.tensorizers import LabelTensorizer
from pytext.data.utils import Vocabulary
from pytext.loss import CrossEntropyLoss
from pytext.models.output_layers.doc_classification_output_layer import (
    ClassificationOutputLayer,
)
from pytext.models.output_layers.intent_slot_output_layer import IntentSlotOutputLayer
from pytext.models.output_layers.word_tagging_output_layer import (
    CRFOutputLayer,
    WordTaggingOutputLayer,
)


class OutputLayerTest(hu.HypothesisTestCase):
    def test_doc_classification_output_layer(self):
        tensorizer = LabelTensorizer()
        tensorizer.vocab = Vocabulary([SpecialTokens.PAD, "foo", "bar"])
        layer = ClassificationOutputLayer.from_config(
            config=ClassificationOutputLayer.Config(loss=CrossEntropyLoss.Config()),
            labels=tensorizer.vocab,
        )
        self.assertEqual(layer.loss_fn.ignore_index, 0)

        # use default pad
        tensorizer.vocab = Vocabulary(["foo", "bar"])
        layer = ClassificationOutputLayer.from_config(
            config=ClassificationOutputLayer.Config(loss=CrossEntropyLoss.Config()),
            labels=tensorizer.vocab,
        )
        self.assertEqual(layer.loss_fn.ignore_index, -1)

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
        vocab = Vocabulary(
            [OutputLayerTest._generate_random_string() for _ in range(num_labels)]
        )

        word_layer = WordTaggingOutputLayer.from_config(
            config=WordTaggingOutputLayer.Config(), labels=vocab
        )
        crf_layer = CRFOutputLayer.from_config(
            config=CRFOutputLayer.Config(), labels=vocab
        )

        logits, seq_lens_tensor = OutputLayerTest._generate_word_tagging_inputs(
            batch_size, num_labels, seq_lens
        )
        context = {"seq_lens": seq_lens_tensor}

        torchsript_word_layer = word_layer.torchscript_predictions()
        torchscript_crf_layer = crf_layer.torchscript_predictions()

        self._validate_word_tagging_result(
            word_layer.get_pred(logits, None, context)[1],
            torchsript_word_layer(logits, context),
            vocab,
        )
        self._validate_word_tagging_result(
            crf_layer.get_pred(logits, None, context)[1],
            torchscript_crf_layer(logits, context),
            vocab,
        )

    @given(
        num_doc_labels=st.integers(2, 6),
        num_word_labels=st.integers(2, 6),
        seq_lens=st.lists(
            elements=st.integers(min_value=4, max_value=10), min_size=1, max_size=10
        ),
    )
    def test_torchscript_intent_slot_output_layer(
        self, num_doc_labels, num_word_labels, seq_lens
    ):
        batch_size = len(seq_lens)
        doc_vocab = Vocabulary(
            [OutputLayerTest._generate_random_string() for _ in range(num_doc_labels)]
        )
        word_vocab = Vocabulary(
            [OutputLayerTest._generate_random_string() for _ in range(num_word_labels)]
        )
        intent_slot_output_layer = IntentSlotOutputLayer.from_config(
            config=IntentSlotOutputLayer.Config(),
            doc_labels=doc_vocab,
            word_labels=word_vocab,
        )
        doc_logits = OutputLayerTest._generate_doc_classification_inputs(
            batch_size, num_doc_labels
        )
        word_logits, seq_lens_tensor = OutputLayerTest._generate_word_tagging_inputs(
            batch_size, num_word_labels, seq_lens
        )
        context = {"seq_lens": seq_lens_tensor}
        torchscript_output_layer = intent_slot_output_layer.torchscript_predictions()

        pt_output = intent_slot_output_layer.get_pred(
            (doc_logits, word_logits), None, context
        )[1]
        ts_output = torchscript_output_layer((doc_logits, word_logits), context)

        self._validate_doc_classification_result(pt_output[0], ts_output[0], doc_vocab)
        self._validate_word_tagging_result(pt_output[1], ts_output[1], word_vocab)

        (
            word_bpe_logits,
            seq_lens_tensor,
            token_indices_tensor,
        ) = OutputLayerTest._generate_bpe_tagging_inputs(
            batch_size, num_word_labels, seq_lens
        )
        context = {"seq_lens": seq_lens_tensor, "token_indices": token_indices_tensor}
        pt_output = intent_slot_output_layer.get_pred(
            (doc_logits, word_bpe_logits), None, context
        )[1]
        ts_output = torchscript_output_layer((doc_logits, word_bpe_logits), context)

        self._validate_doc_classification_result(pt_output[0], ts_output[0], doc_vocab)
        self._validate_word_tagging_result(pt_output[1], ts_output[1], word_vocab)

    @staticmethod
    def _generate_random_string(min_size: int = 2, max_size: int = 8) -> str:
        size = random.randint(min_size, max_size)
        return "".join([random.choice(string.ascii_lowercase) for _ in range(size)])

    @staticmethod
    def _generate_word_tagging_inputs(bsize: int, num_labels: int, seq_lens: List[int]):
        max_seq_length = max(seq_lens)
        logits = torch.randn(bsize, max_seq_length, num_labels)
        return logits, torch.tensor(seq_lens, dtype=torch.int)

    @staticmethod
    def _generate_bpe_tagging_inputs(bsize: int, num_labels: int, seq_lens: List[int]):
        max_seq_length = max(seq_lens)
        max_bpe_length = max_seq_length * 2 + random.randint(
            1, max_seq_length
        )  # Arbitrary length greater than max_seq_length
        logits = torch.randn(bsize, max_bpe_length, num_labels)
        token_begin_indices = []
        for l in seq_lens:
            token_begin_indices.append(
                sorted(random.sample(range(max_bpe_length), l))
                + [0] * (max_seq_length - l)
            )
        return (
            logits,
            torch.tensor(seq_lens, dtype=torch.int),
            torch.tensor(token_begin_indices, dtype=torch.long),
        )

    @staticmethod
    def _generate_doc_classification_inputs(bsize: int, num_labels: int):
        return torch.randn(bsize, num_labels)

    def _validate_doc_classification_result(
        self,
        scores: torch.Tensor,
        ts_results: List[Dict[str, float]],
        vocab: Vocabulary,
    ):
        bsize, num_labels = scores.size()
        self.assertEqual(
            len(ts_results),
            bsize,
            "Batch size must match for pytorch and torchscript class",
        )
        self.assertEqual(
            len(ts_results[0]),
            num_labels,
            "Number of labels must match for pytorch and torchscript class",
        )
        self.assertEqual(
            len(ts_results[0]),
            len(vocab),
            "Number of labels should be same as vocab length",
        )
        for i in range(bsize):
            for label, score in ts_results[i].items():
                self.assertAlmostEqual(
                    score,
                    scores[i][vocab.idx[label]],
                    (
                        "Scores for [{}][{}] element must match for "
                        "pytorch and torchscript class"
                    ).format(i, vocab.idx[label]),
                )

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

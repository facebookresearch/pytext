#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import random
import string
import unittest
from typing import List, Tuple, Optional

import torch
from pytext.torchscript.module import PyTextEmbeddingModule
from pytext.torchscript.tensorizer.tensorizer import ScriptTensorizer, VocabLookup
from pytext.torchscript.utils import (
    pad_2d,
    pad_2d_mask,
    ScriptBatchInput,
)
from pytext.torchscript.vocab import ScriptVocabulary
from torch import Tensor


class MyTensorizer(ScriptTensorizer):
    def __init__(
        self,
        tokenizer: torch.jit.ScriptModule,
        vocab: ScriptVocabulary,
        max_seq_len: int = 100,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.vocab_lookup = VocabLookup(vocab)
        self.max_seq_len = torch.jit.Attribute(max_seq_len, int)

    @torch.jit.script_method
    def _lookup_tokens(self, tokens: List[Tuple[str, int, int]]) -> List[int]:
        return self.vocab_lookup(
            tokens,
            bos_idx=None,
            eos_idx=self.vocab.eos_idx,
            use_eos_token_for_bos=False,
            max_seq_len=self.max_seq_len,
        )[0]

    @torch.jit.script_method
    def _wrap_numberized_tokens(self, token_ids: List[int], idx: int) -> List[int]:
        if idx == 0:
            token_ids = [self.vocab.bos_idx] + token_ids
        return token_ids

    @torch.jit.script_method
    def tokenize(
        self, text_row: Optional[List[str]], token_row: Optional[List[List[str]]]
    ) -> List[List[Tuple[str, int, int]]]:
        """
        Process a single line of raw inputs into tokens, it supports
        two input formats:
            1) a single line of texts (single sentence or a pair)
            2) a single line of pre-processed tokens (single sentence or a pair)
        """
        per_sentence_tokens: List[List[Tuple[str, int, int]]] = []

        if text_row is not None:
            for text in text_row:
                per_sentence_tokens.append(self.tokenizer.tokenize(text))
        elif token_row is not None:
            for sentence_raw_tokens in token_row:
                sentence_tokens: List[Tuple[str, int, int]] = []
                for raw_token in sentence_raw_tokens:
                    sentence_tokens.extend(self.tokenizer.tokenize(raw_token))
                per_sentence_tokens.append(sentence_tokens)

        return per_sentence_tokens

    @torch.jit.script_method
    def numberize(
        self, text_row: Optional[List[str]], token_row: Optional[List[List[str]]]
    ) -> Tuple[List[int], List[int], int, List[int]]:
        """
        Process a single line of raw inputs into numberized result, it supports
        two input formats:
            1) a single line of texts (single sentence or a pair)
            2) a single line of pre-processed tokens (single sentence or a pair)

        This function should handle the logic of calling tokenize(), add special
        tokens and vocab lookup.
        """
        token_ids: List[int] = []
        segment_labels: List[int] = []
        seq_len: int = 0
        positions: List[int] = []
        per_sentence_tokens: List[List[Tuple[str, int, int]]] = self.tokenize(
            text_row, token_row
        )

        for idx, per_sentence_token in enumerate(per_sentence_tokens):
            lookup_ids: List[int] = self._lookup_tokens(per_sentence_token)
            lookup_ids = self._wrap_numberized_tokens(lookup_ids, idx)

            token_ids.extend(lookup_ids)
            segment_labels.extend([idx] * len(lookup_ids))
        seq_len = len(token_ids)
        positions = list(range(seq_len))

        return token_ids, segment_labels, seq_len, positions

    @torch.jit.script_method
    def tensorize(
        self,
        texts: Optional[List[List[str]]] = None,
        tokens: Optional[List[List[List[str]]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process raw inputs into model input tensors, it supports two input
        formats:
            1) multiple rows of texts (single sentence or a pair)
            2) multiple rows of pre-processed tokens (single sentence or a pair)

        This function should handle the logic of calling numberize() and also
        padding the numberized result.
        """
        tokens_2d: List[List[int]] = []
        segment_labels_2d: List[List[int]] = []
        seq_len_2d: List[int] = []
        positions_2d: List[List[int]] = []

        for idx in range(self.batch_size(texts, tokens)):
            numberized: Tuple[List[int], List[int], int, List[int]] = self.numberize(
                self.get_texts_by_index(texts, idx),
                self.get_tokens_by_index(tokens, idx),
            )
            tokens_2d.append(numberized[0])
            segment_labels_2d.append(numberized[1])
            seq_len_2d.append(numberized[2])
            positions_2d.append(numberized[3])

        tokens, pad_mask = pad_2d_mask(tokens_2d, pad_value=self.vocab.pad_idx)
        segment_labels = torch.tensor(
            pad_2d(segment_labels_2d, seq_lens=seq_len_2d, pad_idx=self.vocab.pad_idx),
            dtype=torch.long,
        )
        positions = torch.tensor(
            pad_2d(positions_2d, seq_lens=seq_len_2d, pad_idx=self.vocab.pad_idx),
            dtype=torch.long,
        )

        if self.device == "":
            return tokens, pad_mask, segment_labels, positions
        else:
            return (
                tokens.to(self.device),
                pad_mask.to(self.device),
                segment_labels.to(self.device),
                positions.to(self.device),
            )

    @torch.jit.script_method
    def forward(
        self,
        inputs: ScriptBatchInput,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if inputs.texts is not None or inputs.tokens is not None:
            return self.tensorize(inputs.texts, inputs.tokens)
        else:
            raise RuntimeError("Empty input for both texts and tokens.")


class PytextembeddingmoduleTest(unittest.TestCase):
    def _mock_vocab(self):
        # mapping of vocab index to token is 0-9
        return ScriptVocabulary(
            [str(i) for i in range(0, 10)],
            pad_idx=-1,
            bos_idx=0,
            unk_idx=-1,
        )

    def _mock_tokenizer(self):
        # simple tokenizer
        class MockTokenizer(torch.jit.ScriptModule):
            @torch.jit.script_method
            def tokenize(self, text: str) -> List[Tuple[str, int, int]]:
                return [(s, -1, -1) for s in text.lower().split(" ")]

        return MockTokenizer()

    def _mock_model(self):
        # simple model, return inputs
        class MockModel(torch.jit.ScriptModule):
            def forward(
                self, inputs: Tuple[Tensor, Tensor, Tensor, Tensor]
            ) -> torch.Tensor:
                return inputs[0]

        return MockModel()

    def setUp(self) -> None:
        self.batch_size = 10
        self.NONE_INPUT = None
        self.EMPTYLIST = []
        self.EMPTY_TUPLE = [()]
        self.EMPTY_TUPLE_LIST = [([],)]
        self.ONE_TEXT = [
            ([["1 2 3"]]),
        ]
        self.MULTI_TEXT = [
            ([["1 2 3", "4 5 6"]]),
        ]
        self.MULTI_TEXT_TOO_LONG = [
            ([["1 2 3 4 5 6 7", "4 5 6"]]),
        ]
        self.MULTI_BATCH_MULTI_TEXT = [
            ([["1 2 3", "4 5 6"]]),
            ([["7 8 9", "10 11 12"]]),
        ]
        self.MULTI_BATCH_ONEEMPTY = [
            ([["1 2 3", "4 5 6"]]),
            ([[]]),
        ]
        self.MULTI_BATCH_DIFF_LENGTH = [
            ([["1 2 3", "4 5 6"]]),
            ([["1 2 3", "4 5 6", "7 8 9"]]),
        ]
        self.MULTI_BATCH_DIFF_TEXT_LENGTH = [
            ([["1 2 3", "4 5 6 7 8 9"]]),
            ([["1 2 3", "4 5 6", "7 8 9"]]),
        ]
        self.ONE_TOKEN = [([["token_1_1", "token_2_2"]])]

        vocab = self._mock_vocab()
        tokenizer = self._mock_tokenizer()
        model = self._mock_model()
        tensorizer = MyTensorizer(tokenizer, vocab, max_seq_len=5)

        self.module = PyTextEmbeddingModule(model, tensorizer)

    def test_make_prediction_none(self) -> None:
        # Negative Case: None as input
        with self.assertRaises(RuntimeError):
            self.module.make_prediction(self.NONE_INPUT)

    def test_make_prediction_emptylist(self) -> None:
        # Negative Case: Empty request batch test in List[] format
        with self.assertRaises(torch.jit.Error):
            self.module.make_prediction(self.EMPTYLIST)

    def test_make_prediction_empty_tuple(self) -> None:
        # Negative Case: Bad batch token format with List[Tuple()]
        with self.assertRaises(RuntimeError):
            self.module.make_prediction(self.EMPTY_TUPLE)

    def test_make_prediction_empty_tuple_list(self) -> None:
        # Negative Case: Empty request batch test in List[Tuple([List[]])] format
        with self.assertRaises(torch.jit.Error):
            self.module.make_prediction(self.EMPTY_TUPLE_LIST)

    def test_make_prediction_one_text(self) -> None:
        # Positve Case: list of one text
        outputs = [
            tensor.tolist() for tensor in self.module.make_prediction(self.ONE_TEXT)
        ]
        self.assertEqual(
            outputs,
            [[[0, 1, 2, 3]]],
        )

    def test_make_prediction_multi_text(self) -> None:
        # Positve Case: list of two text
        outputs = [
            tensor.tolist() for tensor in self.module.make_prediction(self.MULTI_TEXT)
        ]
        self.assertEqual(
            outputs,
            [
                [[0, 1, 2, 3], [0, 4, 5, 6]],
            ],
        )

    def test_make_prediction_multi_text_too_long(self) -> None:
        # Positve Case: text length out of 5
        outputs = [
            tensor.tolist()
            for tensor in self.module.make_prediction(self.MULTI_TEXT_TOO_LONG)
        ]
        self.assertEqual(
            outputs,
            [
                [[0, 1, 2, 3, 4, 5], [0, 4, 5, 6, -1, -1]],
            ],
        )

    def test_make_prediction_multi_batch_multi_text(self) -> None:
        # Positive Case: multi-batch of texts
        outputs = [
            tensor.tolist()
            for tensor in self.module.make_prediction(self.MULTI_BATCH_MULTI_TEXT)
        ]
        self.assertEqual(
            outputs,
            [
                [[0, 1, 2, 3], [0, 4, 5, 6]],
                [[0, 7, 8, 9], [0, -1, -1, -1]],
            ],
        )

    def test_make_prediction_multi_batch_oneempty(self) -> None:
        # Positive case: multi-batch, one batch is empty
        outputs = [
            tensor.tolist()
            for tensor in self.module.make_prediction(self.MULTI_BATCH_ONEEMPTY)
        ]
        self.assertEqual(
            outputs,
            [
                [[0, 1, 2, 3], [0, 4, 5, 6]],
                [],
            ],
        )

    def test_make_prediction_multi_batch_diff_length(self) -> None:
        # Positive Case: different numbers of texts in each batch
        outputs = [
            tensor.tolist()
            for tensor in self.module.make_prediction(self.MULTI_BATCH_DIFF_LENGTH)
        ]
        self.assertEqual(
            outputs,
            [
                [[0, 1, 2, 3], [0, 4, 5, 6]],
                [[0, 1, 2, 3], [0, 4, 5, 6], [0, 7, 8, 9]],
            ],
        )

    def test_make_prediction_multi_batch_diff_textlength(self) -> None:
        # Positive Case: padding for different lengths of texts
        outputs = [
            tensor.tolist()
            for tensor in self.module.make_prediction(self.MULTI_BATCH_DIFF_TEXT_LENGTH)
        ]
        self.assertEqual(
            outputs,
            [
                [[0, 1, 2, 3, -1, -1], [0, 4, 5, 6, 7, 8]],
                [[0, 1, 2, 3, -1, -1], [0, 4, 5, 6, -1, -1], [0, 7, 8, 9, -1, -1]],
            ],
        )

    def test_make_prediction_one_token(self) -> None:
        # Positve Case:
        outputs = [
            tensor.tolist() for tensor in self.module.make_prediction(self.ONE_TOKEN)
        ]
        self.assertEqual(
            outputs,
            [
                [[0, -1], [0, -1]],
            ],
        )

    def test_make_batch_invalid_input_list(self) -> None:
        # case: raises a runtime error when input list is invalid
        with self.assertRaises(RuntimeError):
            self.module.make_batch(None, {})

    def test_make_batch_empty_input_list(self) -> None:
        # case: raises a runtime error when input list is empty
        with self.assertRaises(torch.jit.Error):
            self.module.make_batch([], {})

    def test_make_batch_empty_input_tuple(self) -> None:
        # case: raises a runtime error when input tuple is invalid
        with self.assertRaises(RuntimeError):
            self.module.make_batch([()], {})

    def test_make_batch_input_tuple_with_no_position(self) -> None:
        # case: raises a runtime error when input tuple has no position field
        with self.assertRaises(RuntimeError):
            self.module.make_batch([(["1 2"])], {})

    def test_make_batch_with_valid_and_invalid_input_tuples(self) -> None:
        # case: raises a runtime error when input has both valid and invalid tuples
        mega_batch_1 = [(["1 2"], 1), (["3 4"])]  # second tuple has no position
        with self.assertRaises(RuntimeError):
            self.module.make_batch(mega_batch_1, {})

    def test_make_batch_input_tuple_with_empty_list(self) -> None:
        # case: a mega-batch with exactly one tuple (w/ no text) returns one batch
        mega_batch = [([], 1)]
        batches = self.module.make_batch(mega_batch, {})
        self.assertEqual(batches, [mega_batch])

    def test_make_batch_with_one_valid_input_tuple(self) -> None:
        # case: a mega-batch with exactly one tuple returns one batch
        mega_batch = [(["1 2"], 1)]
        batches = self.module.make_batch(mega_batch, {})
        self.assertEqual(batches, [mega_batch])

    def test_make_batch_with_one_valid_input_tuple_with_multi_text(self) -> None:
        # case: a mega-batch with exactly one tuple (w/ multi_text) returns one batch
        mega_batch = [(["1 2 3", "4 5 6"], 1)]
        batches = self.module.make_batch(mega_batch, {})
        self.assertEqual(batches, [mega_batch])

    def test_make_batch_returns_one_batch_for_input_mega_batch(self) -> None:
        # case: a mega-batch with at most 'batch_size' input tuples returns one batch
        mega_batch = []
        for _i in range(self.batch_size):
            mega_batch.append((["1 2 3", "4 5 6"], _i))
        batches = self.module.make_batch(mega_batch, {})
        self.assertEqual(batches, [mega_batch])

    def test_make_batch_returns_multiple_batches_for_input_mega_batch(self) -> None:
        # case: multiple batches are returned for a large input mega-batch
        mega_batch = []
        for _i in range(self.batch_size * 3 + self.batch_size - 1):
            mega_batch.append((["1 2 3", "4 5 6"], _i))
        batches = self.module.make_batch(mega_batch, {})
        start = 0
        while start < min(start + self.batch_size, len(mega_batch)):
            end = min(start + self.batch_size, len(mega_batch))
            self.assertEqual(batches[start // self.batch_size], mega_batch[start:end])
            start = end

    def test_make_batch_returns_one_batch_with_input_tuples_sorted(self) -> None:
        # case: input tuples are returned in a sorted order
        mega_batch = []
        for _i in range(self.batch_size):
            text = self.get_random_string_with_n_tokens(self.batch_size - _i)
            mega_batch.append(([text], _i))
        sorted_mega_batch = sorted(mega_batch, key=lambda x: x[1], reverse=True)
        batches = self.module.make_batch(mega_batch, {})
        self.assertEqual(batches, [sorted_mega_batch])

    def test_make_batch_returns_multiple_batches_with_input_tuples_sorted(self) -> None:
        # case: multiple batches are returned (w/ tuples sorted) for a large input mega-batch
        mega_batch = []
        mega_batch_len = (self.batch_size * 3) + self.batch_size - 1
        # make a large mega-batch so that multiuple batches are returned
        for _i in range(mega_batch_len):
            # each input has multiple text fields of varying lengths
            multi_text = []
            # first add a few short strings
            for short_text_len in range(5):
                multi_text.append(self.get_random_string_with_n_tokens(short_text_len))
            # add a string with many tokens (to verify sorted order)
            multi_text.append(
                self.get_random_string_with_n_tokens(mega_batch_len * 2 - _i)
            )
            mega_batch.append((multi_text, _i))
        sorted_mega_batch = sorted(mega_batch, key=lambda x: x[1], reverse=True)
        batches = self.module.make_batch(mega_batch, {})
        start = 0
        while start < min(start + self.batch_size, len(mega_batch)):
            end = min(start + self.batch_size, len(mega_batch))
            self.assertEqual(
                batches[start // self.batch_size], sorted_mega_batch[start:end]
            )
            start = end

    def get_random_string_with_n_tokens(self, n) -> None:
        return " ".join(random.choice(string.ascii_uppercase) for _ in range(n))

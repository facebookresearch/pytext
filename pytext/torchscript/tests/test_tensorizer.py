#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import random
import unittest
from typing import List, Tuple

import torch
from pytext.torchscript.tensorizer import (
    ScriptBERTTensorizer,
    ScriptRoBERTaTensorizer,
    ScriptRoBERTaTensorizerWithIndices,
    ScriptXLMTensorizer,
)
from pytext.torchscript.tensorizer.tensorizer import VocabLookup
from pytext.torchscript.tokenizer import ScriptDoNothingTokenizer
from pytext.torchscript.tokenizer.tokenizer import ScriptTokenizerBase
from pytext.torchscript.utils import squeeze_1d, squeeze_2d
from pytext.torchscript.vocab import ScriptVocabulary


class TensorizerTest(unittest.TestCase):
    def _mock_vocab(self):
        # mapping of vocab index to token is x: x + 100
        return ScriptVocabulary(
            [str(i) for i in range(100, 303)], pad_idx=200, bos_idx=201, eos_idx=202
        )

    def _mock_tokenizer(self):
        class MockTokenizer(ScriptTokenizerBase):
            def __init__(self, tokens: List[Tuple[str, int, int]]):
                super().__init__()
                self.tokens = torch.jit.Attribute(tokens, List[Tuple[str, int, int]])

            @torch.jit.script_method
            def tokenize(self, text: str) -> List[Tuple[str, int, int]]:
                return self.tokens

        rand_tokens = [(str(random.randint(100, 200)), -1, -1) for i in range(20)]
        return MockTokenizer(rand_tokens), rand_tokens

    def test_lookup_tokens(self):
        _, rand_tokens = self._mock_tokenizer()
        vocab = self._mock_vocab()
        vocab_lookup = VocabLookup(vocab)
        token_ids, start_idxs, end_idxs = vocab_lookup(rand_tokens)

        for token_id, token in zip(token_ids, rand_tokens):
            self.assertEqual(token_id, int(token[0]) - 100)

    def test_lookup_tokens_with_bos_eos(self):
        _, rand_tokens = self._mock_tokenizer()
        vocab = self._mock_vocab()
        vocab_lookup = VocabLookup(vocab)
        token_ids, start_idxs, end_idxs = vocab_lookup(
            rand_tokens, bos_idx=201, eos_idx=202
        )
        self.assertEqual(token_ids[0], 201)
        self.assertEqual(token_ids[-1], 202)
        for token_id, token in zip(token_ids[1:-1], rand_tokens):
            self.assertEqual(token_id, int(token[0]) - 100)

    def test_bert_tensorizer(self):
        tokenizer, rand_tokens = self._mock_tokenizer()
        vocab = self._mock_vocab()

        bert = ScriptBERTTensorizer(tokenizer, vocab, max_seq_len=100)
        token_ids, _, _, _ = bert.numberize(["mock test"], None)
        self.assertEqual(token_ids[0], 201)
        self.assertEqual(token_ids[-1], 202)
        for token_id, token in zip(token_ids[1:-1], rand_tokens):
            self.assertEqual(token_id, int(token[0]) - 100)

    def test_roberta_tensorizer(self):
        tokenizer, rand_tokens = self._mock_tokenizer()
        vocab = self._mock_vocab()

        roberta = ScriptRoBERTaTensorizer(tokenizer, vocab, max_seq_len=100)
        token_ids, _, _, _ = roberta.numberize(["mock test"], None)
        self.assertEqual(token_ids[0], 201)
        self.assertEqual(token_ids[-1], 202)
        for token_id, token in zip(token_ids[1:-1], rand_tokens):
            self.assertEqual(token_id, int(token[0]) - 100)

    def validate_padding(
        self,
        output_tensor: torch.Tensor,
        pad_val: int,
        significant_idxs: List[int],
        expected_batch_size: int,
        expected_token_padding: List[int],
    ):
        output_list = output_tensor.tolist()
        self.assertEqual(len(output_list), expected_batch_size)
        for i in range(expected_batch_size):
            # indices that store significant values
            actual_idxs = significant_idxs[i]
            # create a list of size(expected_padding) filled with the value of pad_idx
            expected_padding_list = [pad_val] * expected_token_padding[i]
            # slice the given output_list from the last substantive index
            actual_padding_list = output_list[i][actual_idxs:]
            self.assertEqual(expected_padding_list, actual_padding_list)

    def get_rand_tokens(self, sizes: List[int]):
        """
        Returns a List[List[int]] of values within range of the vocab
        """
        rand_tokens = []
        for val in sizes:
            rand_tokens.append([str(random.randint(100, 200)) for i in range(val)])
        return rand_tokens

    def _mock_roberta_tensorizer(self, max_seq_len=100):
        return ScriptRoBERTaTensorizerWithIndices(
            tokenizer=ScriptDoNothingTokenizer(),
            vocab=self._mock_vocab(),
            max_seq_len=max_seq_len,
        )

    def _mock_xlm_tensorizer(self, max_seq_len=256):
        return ScriptXLMTensorizer(
            tokenizer=ScriptDoNothingTokenizer(),
            token_vocab=self._mock_vocab(),
            language_vocab=ScriptVocabulary(["ar", "cn", "en"]),
            max_seq_len=256,
            default_language="en",
        )

    def test_roberta_tensorizer_default_padding(self):
        roberta = self._mock_roberta_tensorizer()
        rand_tokens = self.get_rand_tokens([20, 5, 15])

        start_placeholder = 1
        end_placeholder = 1
        # num idxs that store significant values for elem in rand_token, i.e. [22, 7, 17]
        sig_idxs = [start_placeholder + len(t) + end_placeholder for t in rand_tokens]
        # pad every token to bottleneck value, i.e. [0, 15, 5]
        expected_token_padding = [max(sig_idxs) - num for num in sig_idxs]

        tokens, pad_mask, start_indices, end_indices, positions = roberta.tensorize(
            tokens=squeeze_2d(rand_tokens)
        )

        padding_key = {
            tokens: 200,
            pad_mask: 0,
            start_indices: 0,
            end_indices: 0,
            positions: 0,
        }

        # verify padding
        for output_tensor, pad_val in padding_key.items():
            self.validate_padding(
                output_tensor,
                pad_val,
                significant_idxs=sig_idxs,
                expected_batch_size=len(rand_tokens),
                expected_token_padding=expected_token_padding,
            )

    def test_roberta_tensorizer_sequence_padding(self):
        roberta = self._mock_roberta_tensorizer()
        seq_padding_control = [0, 32, 256]
        roberta.set_padding_control("sequence_length", seq_padding_control)
        rand_tokens = self.get_rand_tokens([20, 5, 15])

        start_placeholder = 1
        end_placeholder = 1
        # num idxs that store significant values for elem in rand_token, i.e. [22, 7, 17]
        sig_idxs = [start_placeholder + len(t) + end_placeholder for t in rand_tokens]

        expected_token_size = 32
        # pad every token to bottleneck value, i.e. [0, 15, 5]
        expected_token_padding = [expected_token_size - num for num in sig_idxs]

        tokens, pad_mask, start_indices, end_indices, positions = roberta.tensorize(
            tokens=squeeze_2d(rand_tokens)
        )

        padding_key = {
            tokens: 200,
            pad_mask: 0,
            start_indices: 0,
            end_indices: 0,
            positions: 0,
        }

        # verify padding
        for output_tensor, pad_val in padding_key.items():
            self.validate_padding(
                output_tensor,
                pad_val,
                significant_idxs=sig_idxs,
                expected_batch_size=len(rand_tokens),
                expected_token_padding=expected_token_padding,
            )

    def test_roberta_tensorizer_batch_padding(self):
        roberta = self._mock_roberta_tensorizer()
        batch_padding_control = [0, 3, 6]
        roberta.set_padding_control("batch_length", batch_padding_control)

        rand_tokens = self.get_rand_tokens([25, 15, 5, 30])
        expected_batch_size = 6
        expected_token_size = 32

        tokens, pad_mask, start_indices, end_indices, positions = roberta.tensorize(
            tokens=squeeze_2d(rand_tokens)
        )

        # [27, 17, 7, 32, 0, 0]
        sig_idxs = [1 + len(t) + 1 for t in rand_tokens] + [0, 0]
        # [5, 15, 25, 0, 32, 32]
        expected_token_padding = [expected_token_size - num for num in sig_idxs] + [
            32,
            32,
        ]

        padding_key = {
            tokens: 200,
            pad_mask: 0,
            start_indices: 0,
            end_indices: 0,
            positions: 0,
        }

        # verify padding
        for output_tensor, pad_val in padding_key.items():
            self.validate_padding(
                output_tensor,
                pad_val,
                significant_idxs=sig_idxs,
                expected_batch_size=expected_batch_size,
                expected_token_padding=expected_token_padding,
            )

    def test_roberta_tensorizer_sequence_batch_padding(self):
        roberta = self._mock_roberta_tensorizer()
        seq_padding_control = [0, 48, 256]
        batch_padding_control = [0, 3, 6]
        roberta.set_padding_control("batch_length", batch_padding_control)
        roberta.set_padding_control("sequence_length", seq_padding_control)

        rand_tokens = self.get_rand_tokens([25, 15, 5, 30])
        expected_batch_size = 6
        expected_token_size = 48

        tokens, pad_mask, start_indices, end_indices, positions = roberta.tensorize(
            tokens=squeeze_2d(rand_tokens)
        )

        sig_idxs = [1 + len(t) + 1 for t in rand_tokens] + [0, 0]
        expected_token_padding = [expected_token_size - num for num in sig_idxs] + [
            48,
            48,
        ]

        padding_key = {
            tokens: 200,
            pad_mask: 0,
            start_indices: 0,
            end_indices: 0,
            positions: 0,
        }

        # verify padding
        for output_tensor, pad_val in padding_key.items():
            self.validate_padding(
                output_tensor,
                pad_val,
                significant_idxs=sig_idxs,
                expected_batch_size=expected_batch_size,
                expected_token_padding=expected_token_padding,
            )

    def test_roberta_tensorizer_input_exceeds_max_seq_len(self):
        roberta = self._mock_roberta_tensorizer(max_seq_len=28)

        rand_tokens = self.get_rand_tokens([25, 15, 5, 30])
        expected_batch_size = 4
        expected_token_size = 28

        tokens, pad_mask, start_indices, end_indices, positions = roberta.tensorize(
            tokens=squeeze_2d(rand_tokens)
        )

        sig_idxs = [1 + len(t) + 1 for t in rand_tokens]
        expected_token_padding = [expected_token_size - num for num in sig_idxs]

        padding_key = {
            tokens: 200,
            pad_mask: 0,
            start_indices: 0,
            end_indices: 0,
            positions: 0,
        }

        # verify padding
        for output_tensor, pad_val in padding_key.items():
            self.validate_padding(
                output_tensor,
                pad_val,
                significant_idxs=sig_idxs,
                expected_batch_size=expected_batch_size,
                expected_token_padding=expected_token_padding,
            )

    def test_roberta_tensorizer_seq_padding_size_exceeds_max_seq_len(self):
        roberta = self._mock_roberta_tensorizer(max_seq_len=20)
        seq_padding_control = [0, 32, 256]
        roberta.set_padding_control("sequence_length", seq_padding_control)

        rand_tokens = self.get_rand_tokens([30, 20, 10])

        tokens, pad_mask, start_indices, end_indices, positions = roberta.tensorize(
            tokens=squeeze_2d(rand_tokens)
        )

        sig_idxs = [len(t) + 2 for t in rand_tokens]
        expected_batch_size = 3
        expected_token_size = min(
            max(max(sig_idxs), seq_padding_control[1]), roberta.max_seq_len
        )
        expected_token_padding = [max(0, expected_token_size - cnt) for cnt in sig_idxs]
        sig_idxs = [expected_token_size - cnt for cnt in expected_token_padding]

        padding_key = {
            tokens: 200,
            pad_mask: 0,
            start_indices: 0,
            end_indices: 0,
            positions: 0,
        }

        # verify padding
        for output_tensor, pad_val in padding_key.items():
            self.validate_padding(
                output_tensor,
                pad_val,
                significant_idxs=sig_idxs,
                expected_batch_size=expected_batch_size,
                expected_token_padding=expected_token_padding,
            )

    def test_xlm_token_tensorizer(self):
        xlm = self._mock_xlm_tensorizer()
        rand_tokens = self.get_rand_tokens([20, 10])

        tokens, pad_masks, languages, positions = xlm.tensorize(
            tokens=squeeze_2d(rand_tokens)
        )
        tokens = tokens.tolist()
        # eos token
        self.assertEqual(tokens[0][0], 202)
        self.assertEqual(tokens[0][-1], 202)
        # pad token
        self.assertEqual(tokens[1][12:], [200] * 10)

        languages = languages.tolist()
        self.assertEqual(languages[0], [2] * len(tokens[0]))
        self.assertEqual(languages[1][12:], [0] * 10)

        tokens, pad_masks, languages, positions = xlm.tensorize(
            tokens=squeeze_2d(rand_tokens), languages=squeeze_1d(["cn", "en"])
        )
        languages = languages.tolist()
        self.assertEqual(languages[0][:], [1] * len(tokens[0]))
        self.assertEqual(languages[1][:12], [2] * 12)

    def test_xlm_tensorizer_default_padding(self):
        xlm = self._mock_xlm_tensorizer()
        rand_tokens = self.get_rand_tokens([20, 10])

        tokens, pad_masks, languages, positions = xlm.tensorize(
            tokens=squeeze_2d(rand_tokens)
        )

        sig_idxs = [len(t) + 2 for t in rand_tokens]
        expected_token_size = max(sig_idxs)
        expected_token_padding = [expected_token_size - cnt for cnt in sig_idxs]
        expected_batch_size = len(rand_tokens)

        # verify tensorized tokens padding
        tokens = tokens.tolist()
        self.assertEqual(len(tokens), expected_batch_size)
        self.assertEqual(
            max(len(t) for t in tokens),
            min(len(t) for t in tokens),
            expected_token_size,
        )
        for i in range(expected_batch_size):
            self.assertEqual(
                tokens[i][sig_idxs[i] :], [200] * expected_token_padding[i]
            )

        # verify tensorized languages
        languages = languages.tolist()
        self.assertEqual(len(languages), expected_batch_size)
        for i in range(expected_batch_size):
            self.assertEqual(languages[i][: sig_idxs[i]], [2] * sig_idxs[i])
            self.assertEqual(
                languages[i][sig_idxs[i] :], [0] * expected_token_padding[i]
            )

        # verify tensorized postions
        positions = positions.tolist()
        self.assertEqual(len(positions), expected_batch_size)
        for i in range(expected_batch_size):
            self.assertEqual(
                positions[i][sig_idxs[i] :], [0] * expected_token_padding[i]
            )

        # verify pad_masks
        pad_masks = pad_masks.tolist()
        self.assertEqual(len(pad_masks), expected_batch_size)
        for i in range(expected_batch_size):
            self.assertEqual(pad_masks[i][: sig_idxs[i]], [1] * sig_idxs[i])
            self.assertEqual(
                pad_masks[i][sig_idxs[i] :], [0] * expected_token_padding[i]
            )

    def test_xlm_tensorizer_sequence_padding(self):
        xlm = self._mock_xlm_tensorizer()

        padding_control = [0, 32, 256]
        xlm.set_padding_control("sequence_length", padding_control)

        rand_tokens = self.get_rand_tokens([20, 10])

        tokens, pad_masks, languages, positions = xlm.tensorize(
            tokens=squeeze_2d(rand_tokens),
        )

        sig_idxs = [len(t) + 2 for t in rand_tokens]
        expected_token_size = min(
            max(padding_control[1], max(sig_idxs)), xlm.max_seq_len
        )
        expected_token_padding = [expected_token_size - cnt for cnt in sig_idxs]

        padding_key = {
            tokens: 200,
            pad_masks: 0,
            languages: 0,
            positions: 0,
        }

        # verify padding
        for output_tensor, pad_val in padding_key.items():
            self.validate_padding(
                output_tensor,
                pad_val,
                significant_idxs=sig_idxs,
                expected_batch_size=len(rand_tokens),
                expected_token_padding=expected_token_padding,
            )

    def test_xlm_tensorizer_batch_padding(self):
        xlm = self._mock_xlm_tensorizer()

        batch_padding_control = [0, 3, 6]
        xlm.set_padding_control("batch_length", batch_padding_control)

        rand_tokens = self.get_rand_tokens([20, 10])

        tokens, pad_masks, languages, positions = xlm.tensorize(
            tokens=squeeze_2d(rand_tokens),
        )

        sig_idxs = [len(t) + 2 for t in rand_tokens] + [0]
        expected_token_size = max(sig_idxs)
        expected_batch_size = min(
            max(len(rand_tokens), batch_padding_control[1]), xlm.max_seq_len
        )
        expected_token_padding = [expected_token_size - cnt for cnt in sig_idxs]

        padding_key = {
            tokens: 200,
            pad_masks: 0,
            languages: 0,
            positions: 0,
        }

        # verify padding
        for output_tensor, pad_val in padding_key.items():
            self.validate_padding(
                output_tensor,
                pad_val,
                significant_idxs=sig_idxs,
                expected_batch_size=expected_batch_size,
                expected_token_padding=expected_token_padding,
            )

    def test_xlm_tensorizer_sequence_and_batch_padding(self):
        xlm = self._mock_xlm_tensorizer()

        seq_padding_control = [0, 32, 256]
        xlm.set_padding_control("sequence_length", seq_padding_control)
        batch_padding_control = [0, 3, 6]
        xlm.set_padding_control("batch_length", batch_padding_control)

        rand_tokens = self.get_rand_tokens([20, 10])

        tokens, pad_masks, languages, positions = xlm.tensorize(
            tokens=squeeze_2d(rand_tokens),
        )

        sig_idxs = [len(t) + 2 for t in rand_tokens]
        expected_batch_size = min(
            max(len(rand_tokens), batch_padding_control[1]), xlm.max_seq_len
        )
        sig_idxs += [0] * (expected_batch_size - len(sig_idxs))
        expected_token_size = min(
            max(seq_padding_control[1], max(sig_idxs)), xlm.max_seq_len
        )
        expected_token_padding = [expected_token_size - cnt for cnt in sig_idxs]

        padding_key = {
            tokens: 200,
            pad_masks: 0,
            languages: 0,
            positions: 0,
        }

        # verify padding
        for output_tensor, pad_val in padding_key.items():
            self.validate_padding(
                output_tensor,
                pad_val,
                significant_idxs=sig_idxs,
                expected_batch_size=expected_batch_size,
                expected_token_padding=expected_token_padding,
            )

    def test_xlm_tensorizer_input_sequence_exceeds_max_seq_len(self):
        xlm = self._mock_xlm_tensorizer(max_seq_len=20)
        rand_tokens = self.get_rand_tokens([30, 10])

        tokens, pad_masks, languages, positions = xlm.tensorize(
            tokens=squeeze_2d(rand_tokens),
        )

        sig_idxs = [len(t) + 2 for t in rand_tokens]
        expected_token_size = min(max(sig_idxs), xlm.max_seq_len)
        expected_token_padding = [max(0, expected_token_size - cnt) for cnt in sig_idxs]
        sig_idxs = [expected_token_size - cnt for cnt in expected_token_padding]

        padding_key = {
            tokens: 200,
            pad_masks: 0,
            languages: 0,
            positions: 0,
        }

        # verify padding
        for output_tensor, pad_val in padding_key.items():
            self.validate_padding(
                output_tensor,
                pad_val,
                significant_idxs=sig_idxs,
                expected_batch_size=len(rand_tokens),
                expected_token_padding=expected_token_padding,
            )

    def test_xlm_tensorizer_seq_padding_size_exceeds_max_seq_len(self):
        xlm = self._mock_xlm_tensorizer(max_seq_len=20)
        seq_padding_control = [0, 32, 256]
        xlm.set_padding_control("sequence_length", seq_padding_control)

        rand_tokens = self.get_rand_tokens([30, 20, 10])

        tokens, pad_masks, languages, positions = xlm.tensorize(
            tokens=squeeze_2d(rand_tokens),
        )

        sig_idxs = [len(t) + 2 for t in rand_tokens]
        expected_token_size = min(
            max(max(sig_idxs), seq_padding_control[1]), xlm.max_seq_len
        )
        expected_token_padding = [max(0, expected_token_size - cnt) for cnt in sig_idxs]
        sig_idxs = [expected_token_size - cnt for cnt in expected_token_padding]

        padding_key = {
            tokens: 200,
            pad_masks: 0,
            languages: 0,
            positions: 0,
        }

        # verify padding
        for output_tensor, pad_val in padding_key.items():
            self.validate_padding(
                output_tensor,
                pad_val,
                significant_idxs=sig_idxs,
                expected_batch_size=len(rand_tokens),
                expected_token_padding=expected_token_padding,
            )

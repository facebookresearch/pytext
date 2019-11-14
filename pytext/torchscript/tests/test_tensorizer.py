#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import random
import unittest
from typing import List, Tuple

import torch
from pytext.torchscript.tensorizer import (
    ScriptBERTTensorizer,
    ScriptRoBERTaTensorizer,
    ScriptXLMTensorizer,
)
from pytext.torchscript.tensorizer.tensorizer import VocabLookup
from pytext.torchscript.tokenizer import ScriptDoNothingTokenizer
from pytext.torchscript.tokenizer.tokenizer import ScriptTextTokenizerBase
from pytext.torchscript.utils import squeeze_1d, squeeze_2d
from pytext.torchscript.vocab import ScriptVocabulary


class TensorizerTest(unittest.TestCase):
    def _mock_vocab(self):
        # mapping of vocab index to token is x: x + 100
        return ScriptVocabulary(
            [str(i) for i in range(100, 203)], pad_idx=200, bos_idx=201, eos_idx=202
        )

    def _mock_tokenizer(self):
        class MockTokenizer(ScriptTextTokenizerBase):
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

    def test_xlm_token_tensorizer(self):
        vocab = self._mock_vocab()

        xlm = ScriptXLMTensorizer(
            tokenizer=ScriptDoNothingTokenizer(),
            token_vocab=vocab,
            language_vocab=ScriptVocabulary(["ar", "cn", "en"]),
            max_seq_len=256,
            default_language="en",
        )
        rand_tokens = [
            [str(random.randint(100, 200)) for i in range(20)],
            [str(random.randint(100, 200)) for i in range(10)],
        ]

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

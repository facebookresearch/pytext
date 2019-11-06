#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

from pytext.data.tensorizers import SentencePieceTokenizer


class SentencePieceTokenizerTest(unittest.TestCase):
    def test_tokenize(self):
        sentence = "Testing out sentencepiece"
        expected = [
            '▁T',
            'est',
            'ing',
            '▁out',
            '▁sen',
            't',
            'ence',
            'p',
            'i',
            'e',
            'ce',
        ]
        sp_tokenizer = SentencePieceTokenizer.from_config(
            SentencePieceTokenizer.Config(
                sp_model_path="tests/models/sentencepiece.model"
            )
        )
        tokens = sp_tokenizer.tokenize(sentence)
        tokens = [token.value for token in tokens]
        self.assertEqual(tokens, expected)

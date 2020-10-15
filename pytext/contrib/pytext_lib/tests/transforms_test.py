#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reservedimport unittest

import os
import unittest

import torch
from pytext.common.constants import SpecialTokens
from pytext.contrib.pytext_lib.transforms import TruncateTransform, VocabTransform


class TestTruncateTranform(unittest.TestCase):

    DATA = [[0, 1, 2, 3, 4, 5, 6], [0, 1, 2], []]
    MAX_SEQ_LEN = 4

    def setUp(self):
        self.base_dir = os.path.join(os.path.dirname(__file__), "data")

    def test_truncate_transform(self):
        transform = TruncateTransform(max_seq_len=TestTruncateTranform.MAX_SEQ_LEN)
        res = transform(TestTruncateTranform.DATA)

        for row in res:
            # Truncate lengths above max_seq_len, smaller lens aren't padded.
            self.assertEqual(len(row), min(TestTruncateTranform.MAX_SEQ_LEN, len(row)))

    def test_truncate_transform_torchscript(self):
        transform = TruncateTransform(max_seq_len=TestTruncateTranform.MAX_SEQ_LEN)
        ts_transform = torch.jit.script(transform)
        res = ts_transform(TestTruncateTranform.DATA)

        for row in res:
            # Truncate lengths above max_seq_len, smaller lens aren't padded.
            self.assertEqual(len(row), min(TestTruncateTranform.MAX_SEQ_LEN, len(row)))

    def test_vocab_transform(self):
        transform = VocabTransform(os.path.join(self.base_dir, "vocab_dummy"))
        # <unk> added by fairseq
        tokens = [["<unk>", ",", "."], ["▁que", "▁и", "i", "e"]]
        expected = [[3, 4, 5], [41, 35, 14, 13]]
        self.assertEqual(transform(tokens), expected)

    def test_vocab_transform_replace(self):
        transform = VocabTransform(
            os.path.join(self.base_dir, "vocab_dummy"),
            special_token_replacements={"<unk>": SpecialTokens.UNK},
        )
        # Replace <unk> added by fairseq with our token
        tokens = [["__UNKNOWN__", ",", "."], ["▁que", "▁и", "i", "e"]]
        expected = [[3, 4, 5], [41, 35, 14, 13]]
        self.assertEqual(transform(tokens), expected)

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

    def test_vocab_transform_bos(self):
        transform = VocabTransform(
            os.path.join(self.base_dir, "vocab_dummy"), add_bos=True
        )
        tokens = [["<unk>", ",", "."], ["▁que", "▁и", "i", "e"]]
        expected = [[101, 3, 4, 5], [101, 41, 35, 14, 13]]
        self.assertEqual(transform(tokens), expected)

    def test_vocab_transform_eos(self):
        transform = VocabTransform(
            os.path.join(self.base_dir, "vocab_dummy"), add_eos=True
        )
        tokens = [["<unk>", ",", "."], ["▁que", "▁и", "i", "e"]]
        expected = [[3, 4, 5, 103], [41, 35, 14, 13, 103]]
        self.assertEqual(transform(tokens), expected)

    def test_vocab_transform_bos_and_eos(self):
        transform = VocabTransform(
            os.path.join(self.base_dir, "vocab_dummy"), add_bos=True, add_eos=True
        )
        tokens = [["<unk>", ",", "."], ["▁que", "▁и", "i", "e"]]
        expected = [[101, 3, 4, 5, 103], [101, 41, 35, 14, 13, 103]]
        self.assertEqual(transform(tokens), expected)

    def test_vocab_transform_truncate(self):
        transform = VocabTransform(
            os.path.join(self.base_dir, "vocab_dummy"), max_seq_len=2
        )
        tokens = [["<unk>", ",", "."], ["▁que", "▁и", "i", "e"], ["i"]]
        expected = [[3, 4], [41, 35], [14]]
        self.assertEqual(transform(tokens), expected)

    def test_vocab_transform_truncate_bos(self):
        transform = VocabTransform(
            os.path.join(self.base_dir, "vocab_dummy"), max_seq_len=2, add_bos=True
        )
        # <unk> added by fairseq
        tokens = [["<unk>", ",", "."], ["▁que", "▁и", "i", "e"], ["i"]]
        expected = [[101, 3], [101, 41], [101, 14]]
        self.assertEqual(transform(tokens), expected)

    def test_vocab_transform_truncate_eos(self):
        transform = VocabTransform(
            os.path.join(self.base_dir, "vocab_dummy"), max_seq_len=2, add_eos=True
        )
        tokens = [["<unk>", ",", "."], ["▁que", "▁и", "i", "e"], ["i"]]
        expected = [[3, 103], [41, 103], [14, 103]]
        self.assertEqual(transform(tokens), expected)

    def test_vocab_transform_truncate_bos_and_eos(self):
        transform = VocabTransform(
            os.path.join(self.base_dir, "vocab_dummy"),
            max_seq_len=3,
            add_bos=True,
            add_eos=True,
        )
        tokens = [["<unk>", ",", "."], ["▁que", "▁и", "i", "e"], ["i"]]
        expected = [[101, 3, 103], [101, 41, 103], [101, 14, 103]]
        self.assertEqual(transform(tokens), expected)

    def test_vocab_transform_truncate_bos_and_eos_replace(self):
        """
        Can be easily called as RoBERTa vocab look up test.
        We need BOS = 0 and EOS = 2 for pretrained models compat.
        """
        transform = VocabTransform(
            os.path.join(self.base_dir, "vocab_dummy"),
            max_seq_len=3,
            add_bos=True,
            add_eos=True,
            special_token_replacements={
                "<pad>": SpecialTokens.PAD,
                "<s>": SpecialTokens.BOS,
                "</s>": SpecialTokens.EOS,
                "<unk>": SpecialTokens.UNK,
                "<mask>": SpecialTokens.MASK,
            },
        )
        tokens = [["<unk>", ",", "."], ["▁que", "▁и", "i", "e"], ["i"]]
        expected = [[0, 3, 2], [0, 41, 2], [0, 14, 2]]
        self.assertEqual(transform(tokens), expected)

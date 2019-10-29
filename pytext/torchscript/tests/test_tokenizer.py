#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import io
import pickle
import unittest

import torch
from pytext.torchscript.tokenizer import ScriptBPE
from pytext.torchscript.utils import make_byte_inputs, utf8_chars


BPE_VOCAB_FILE = io.StringIO(
    """
hello_EOW 20
world_EOW 18
th  17
is_EOW 16
bpe_EOW 15
! 14
h 13
t 6
s_EOW 2
i -1
今_EOW -2
"""
)


class BPETest(unittest.TestCase):
    def test_utf8_chars(self):
        words = ["hello", "💩", "¯\\_(ツ)_/¯", "今日"]
        for word in words:
            self.assertEqual(list(word), utf8_chars(word))

    def test_simple_bpe(self):
        BPE_VOCAB_FILE.seek(0)
        bpe = ScriptBPE.from_vocab_file(BPE_VOCAB_FILE)
        tokenized = bpe.tokenize(["hello", "world", "this", "is", "bpe", "今日"])
        self.assertEqual(
            ["hello_EOW", "world_EOW", "th", "is_EOW", "is_EOW", "bpe_EOW", "今_EOW"],
            tokenized,
        )

    def test_pickle_bpe(self):
        BPE_VOCAB_FILE.seek(0)
        original_bpe = ScriptBPE.from_vocab_file(BPE_VOCAB_FILE)
        bpe = pickle.loads(pickle.dumps(original_bpe))
        tokenized = bpe.tokenize(["hello", "world", "this", "is", "bpe", "今日"])
        self.assertEqual(
            ["hello_EOW", "world_EOW", "th", "is_EOW", "is_EOW", "bpe_EOW", "今_EOW"],
            tokenized,
        )

    def test_make_bytes_input(self):
        s1 = "I want some coffee today"
        s2 = "Turn it up"
        max_char_length = 5

        batch = [s1.split(), s2.split()]
        bytes, seq_lens = make_byte_inputs(batch, max_char_length)

        def to_bytes(word, pad_to):
            return list(word.encode()) + [0] * (pad_to - len(word))

        expected_bytes = [
            [
                to_bytes("I", 5),
                to_bytes("want", 5),
                to_bytes("some", 5),
                to_bytes("coffe", 5),
                to_bytes("today", 5),
            ],
            [
                to_bytes("Turn", 5),
                to_bytes("it", 5),
                to_bytes("up", 5),
                to_bytes("", 5),
                to_bytes("", 5),
            ],
        ]
        expected_seq_lens = [5, 3]

        self.assertIsInstance(bytes, torch.LongTensor)
        self.assertIsInstance(seq_lens, torch.LongTensor)
        self.assertEqual(bytes.tolist(), expected_bytes)
        self.assertEqual(seq_lens.tolist(), expected_seq_lens)

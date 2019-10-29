#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

import torch
from pytext.torchscript.vocab import ScriptVocabulary
from torch import jit


class VocabTest(unittest.TestCase):
    def setUp(self):
        vocab_list = ["UNK", "a", "b", "c", "d"]
        self.vocab = ScriptVocabulary(vocab_list)

    def test_vocab_lookup(self):
        # There are bugs with just making this a script, eventually these can be simpler
        class LookupWord(jit.ScriptModule):
            def __init__(self, vocab):
                super().__init__()
                self.vocab = vocab

            @jit.script_method
            def forward(self, word: str):
                return self.vocab.idx[word]

        lookup_word = LookupWord(self.vocab)

        self.assertEqual(1, lookup_word("a"))
        self.assertEqual(3, lookup_word("c"))
        with self.assertRaises(Exception):
            lookup_word("notaword")

    def test_vocab_idx_lookup(self):
        # There are bugs with just making this a script, eventually these can be simpler
        class LookupIndex(jit.ScriptModule):
            def __init__(self, vocab):
                super().__init__()
                self.vocab = vocab

            @jit.script_method
            def forward(self, i: int):
                return self.vocab.vocab[i]

        lookup_idx = LookupIndex(self.vocab)

        self.assertEqual("UNK", lookup_idx(0))
        self.assertEqual("b", lookup_idx(2))
        with self.assertRaises(Exception):
            lookup_idx(20)

    def test_lookup_1d(self):
        self.assertEqual(
            [1, 0, 3, 4], self.vocab.lookup_indices_1d(["a", "e", "c", "d"])
        )
        self.assertEqual([], self.vocab.lookup_indices_1d([]))

    def test_lookup_2d(self):
        self.assertEqual(
            [[1, 0, 3, 4], [], [2]],
            self.vocab.lookup_indices_2d([["a", "e", "c", "d"], [], ["b"]]),
        )
        self.assertEqual([], self.vocab.lookup_indices_2d([]))

    def test_custom_unk(self):
        vocab_list = ["a", "UNK", "b", "c", "d"]
        vocab = ScriptVocabulary(vocab_list, unk_idx=1)
        self.assertEqual([0, 1, 3, 4], vocab.lookup_indices_1d(["a", "e", "c", "d"]))

    def test_lookup_words_1d_cycle_heuristic(self):
        self.assertEqual(
            self.vocab.lookup_words_1d_cycle_heuristic(
                torch.tensor([1, 0, 0]), [], ["y", "z"]
            ),
            ["a", "y", "z"],
        )

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from __future__ import absolute_import, division, print_function, unicode_literals

import unittest
from collections import Counter

from pytext.common.constants import VocabMeta
from pytext.fields.char_field import CharFeatureField
from pytext.utils import precision


pad = VocabMeta.PAD_TOKEN


class CharFieldTest(unittest.TestCase):
    def test_build_vocab(self):
        char_field = CharFeatureField(
            pad_token=VocabMeta.PAD_TOKEN,
            unk_token=VocabMeta.UNK_TOKEN,
            batch_first=True,
            min_freq=2,
        )
        utterances = [
            [
                ["A", "l", "l", "l", pad],
                ["u", "r", pad, pad, pad],
                ["b", "@", "$", "3", "s"],
                ["b", "2", pad, pad, pad],
                ["u", "s", pad, pad, pad],
            ],
            [["p", "l", "a", "y", pad], ["m", "u", "s", "i", "c"]],
        ]
        expected_freqs = Counter(
            {
                pad: 11,
                "l": 4,
                "u": 3,
                "s": 3,
                "b": 2,
                "A": 1,
                "r": 1,
                "@": 1,
                "$": 1,
                "3": 1,
                "2": 1,
                "p": 1,
                "a": 1,
                "y": 1,
                "m": 1,
                "i": 1,
                "c": 1,
            }
        )
        expected_itos = [VocabMeta.UNK_TOKEN, pad, "l", "s", "u", "b"]
        preprocessed_data = [char_field.preprocess(x) for x in utterances]
        char_field.build_vocab(preprocessed_data, min_freq=2)
        self.assertEqual(char_field.vocab.freqs, expected_freqs)
        self.assertEqual(char_field.vocab.itos, expected_itos)

    def test_pad_and_numericalize(self):
        char_field = CharFeatureField(
            pad_token=VocabMeta.PAD_TOKEN,
            unk_token=VocabMeta.UNK_TOKEN,
            batch_first=True,
            max_word_length=4,
        )
        utterances = [
            [
                ["A", "l", "l", "l"],
                ["u", "r", pad, pad],
                ["b", "@", "$", "s"],
                ["b", "2", pad, pad],
                ["u", "s", pad, pad],
            ],
            [["p", "l", "a", "y", pad], ["m", "u", "s", "i", "c"]],
        ]
        # The padded chars should be sorted by descending length. The commented
        # indices are meant to help relate to the expected_stitch_index.
        expected_padded_chars = [
            [
                ["A", "l", "l", "l"],
                ["u", "r", pad, pad],
                ["b", "@", "$", "s"],
                ["b", "2", pad, pad],
                ["u", "s", pad, pad],
            ],
            [
                ["p", "l", "a", "y"],
                ["m", "u", "s", "i"],
                [pad, pad, pad, pad],
                [pad, pad, pad, pad],
                [pad, pad, pad, pad],
            ],
        ]

        minibatch = [char_field.preprocess(x) for x in utterances]
        char_field.build_vocab(minibatch)

        # Begin tests for pad().
        padded_minibatch = char_field.pad(minibatch)
        self.assertEqual(padded_minibatch, expected_padded_chars)

        # Begin tests for numericalize().
        expected_numericalized_chars = [
            [[char_field.vocab.stoi[char] for char in word] for word in sent]
            for sent in expected_padded_chars
        ]
        numericalized_minibatch = char_field.numericalize(
            padded_minibatch, device="cpu"
        )
        self.assertEqual(
            numericalized_minibatch.numpy().tolist(), expected_numericalized_chars
        )

        precision.FP16_ENABLED = True
        padded_minibatch = char_field.pad(minibatch)
        self.assertTrue(len(padded_minibatch[0]) % 8 == 0)
        precision.FP16_ENABLED = False

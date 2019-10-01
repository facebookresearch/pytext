#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

import numpy as np
from pytext.common.constants import VocabMeta
from pytext.fields.dict_field import DictFeatureField
from pytext.utils import precision


FEATS_VOCAB = [[["name"]], [["cities"]], [["time"]]]
# The featurizer supplies dict feats str that are padded appropriately at
# the utterance level; the test should reflect the same too
DICT_FEATS_STR = [
    [
        VocabMeta.PAD_TOKEN,
        VocabMeta.PAD_TOKEN,
        VocabMeta.PAD_TOKEN,
        VocabMeta.PAD_TOKEN,
        "name",
        "cities",
    ],
    [VocabMeta.PAD_TOKEN, "time"],
]
WEIGHTS = [[0.0, 0.0, 0.0, 0.0, 1.0, 1.0], [0.0, 1.0]]
LENGTHS = [[1, 1, 2], [1, 1]]
# right padding
PADDED_DICT_FEATS = [
    [
        VocabMeta.PAD_TOKEN,
        VocabMeta.PAD_TOKEN,
        VocabMeta.PAD_TOKEN,
        VocabMeta.PAD_TOKEN,
        "name",
        "cities",
    ],
    [
        VocabMeta.PAD_TOKEN,
        VocabMeta.PAD_TOKEN,
        "time",
        VocabMeta.PAD_TOKEN,
        VocabMeta.PAD_TOKEN,
        VocabMeta.PAD_TOKEN,
    ],
]

PADDED_DICT_WEIGHTS = [[0.0, 0.0, 0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]
PADDED_LENGTHS = [[1, 1, 2], [1, 1, 1]]
NUMERICAL_FEATS = np.array([[1, 1, 1, 1, 3, 2], [1, 1, 4, 1, 1, 1]])

# left padding
LEFT_PADDED_DICT_FEATS = [
    [
        VocabMeta.PAD_TOKEN,
        VocabMeta.PAD_TOKEN,
        VocabMeta.PAD_TOKEN,
        VocabMeta.PAD_TOKEN,
        "name",
        "cities",
    ],
    [
        VocabMeta.PAD_TOKEN,
        VocabMeta.PAD_TOKEN,
        VocabMeta.PAD_TOKEN,
        VocabMeta.PAD_TOKEN,
        "time",
        VocabMeta.PAD_TOKEN,
    ],
]

LEFT_PADDED_DICT_WEIGHTS = [
    [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
]
LEFT_PADDED_LENGTHS = [[1, 1, 2], [1, 1, 1]]
LEFT_NUMERICAL_FEATS = np.array([[1, 1, 1, 1, 3, 2], [1, 1, 1, 1, 4, 1]])


class DictFieldTest(unittest.TestCase):
    def setUp(self):
        self.dict_field = DictFeatureField(
            batch_first=True,
            pad_token=VocabMeta.PAD_TOKEN,
            unk_token=VocabMeta.UNK_TOKEN,
            left_pad=False,
        )
        self.dict_field.build_vocab(FEATS_VOCAB)
        print(self.dict_field.vocab.stoi)

    def test_right_pad_numericalize(self):
        minibatch = [dict_feat for dict_feat in zip(DICT_FEATS_STR, WEIGHTS, LENGTHS)]
        padded_feats, padded_weights, padded_lengths = self.dict_field.pad(minibatch)

        self.assertEqual(padded_feats, PADDED_DICT_FEATS)
        self.assertEqual(padded_weights, PADDED_DICT_WEIGHTS)
        self.assertEqual(padded_lengths, PADDED_LENGTHS)
        feats, _, _ = self.dict_field.numericalize(
            (padded_feats, padded_weights, padded_lengths), device="cpu"
        )
        np.testing.assert_array_equal(feats.data.numpy(), NUMERICAL_FEATS)

        precision.FP16_ENABLED = True
        padded_feats, padded_weights, padded_lengths = self.dict_field.pad(minibatch)
        self.assertTrue(len(padded_feats[0]) % 8 == 0)
        self.assertTrue(len(padded_weights[0]) % 8 == 0)
        precision.FP16_ENABLED = False

    def test_left_pad_numericalize(self):
        minibatch = [dict_feat for dict_feat in zip(DICT_FEATS_STR, WEIGHTS, LENGTHS)]
        self.dict_field.pad_first = True
        padded_feats, padded_weights, padded_lengths = self.dict_field.pad(minibatch)

        self.assertEqual(padded_feats, LEFT_PADDED_DICT_FEATS)
        self.assertEqual(padded_weights, LEFT_PADDED_DICT_WEIGHTS)
        self.assertEqual(padded_lengths, LEFT_PADDED_LENGTHS)
        feats, _, _ = self.dict_field.numericalize(
            (padded_feats, padded_weights, padded_lengths), device="cpu"
        )
        np.testing.assert_array_equal(feats.data.numpy(), LEFT_NUMERICAL_FEATS)

        precision.FP16_ENABLED = True
        padded_feats, padded_weights, padded_lengths = self.dict_field.pad(minibatch)
        self.assertTrue(len(padded_feats[0]) % 8 == 0)
        self.assertTrue(len(padded_weights[0]) % 8 == 0)
        precision.FP16_ENABLED = False

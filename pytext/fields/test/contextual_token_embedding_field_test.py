#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

import numpy as np
from pytext.fields.contextual_token_embedding_field import ContextualTokenEmbeddingField


class ContextualTokenEmbeddingFieldTest(unittest.TestCase):
    def setUp(self):
        self.field = ContextualTokenEmbeddingField(
            pad_token=None, unk_token=None, batch_first=True
        )
        self.unpadded_batch = [
            [
                [0.1, 0.2, 0.3, 0.4, 0.5],
                [1.1, 1.2, 1.3, 1.4, 1.5],
                [2.1, 2.2, 2.3, 2.4, 2.5],
                [3.1, 3.2, 3.3, 3.4, 3.5],
            ],
            [
                [0.1, 0.2, 0.3, 0.4, 0.5],
                [1.1, 1.2, 1.3, 1.4, 1.5],
                [2.1, 2.2, 2.3, 2.4, 2.5],
            ],
            [[0.1, 0.2, 0.3, 0.4, 0.5], [1.1, 1.2, 1.3, 1.4, 1.5]],
        ]
        self.expected_padded_batch = [
            [
                [0.1, 0.2, 0.3, 0.4, 0.5],
                [1.1, 1.2, 1.3, 1.4, 1.5],
                [2.1, 2.2, 2.3, 2.4, 2.5],
                [3.1, 3.2, 3.3, 3.4, 3.5],
            ],
            [
                [0.1, 0.2, 0.3, 0.4, 0.5],
                [1.1, 1.2, 1.3, 1.4, 1.5],
                [2.1, 2.2, 2.3, 2.4, 2.5],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.1, 0.2, 0.3, 0.4, 0.5],
                [1.1, 1.2, 1.3, 1.4, 1.5],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        ]

    def test_pad(self):
        padded_batch = self.field.pad(self.unpadded_batch)
        expected_padded_batch_np = np.array(self.expected_padded_batch)
        padded_batch_np = np.array(padded_batch)

        self.assertEqual(padded_batch_np.shape, expected_padded_batch_np.shape)
        for i in range(len(self.expected_padded_batch)):
            for j in range(len(self.expected_padded_batch[i])):
                for k in range(len(self.expected_padded_batch[i][j])):
                    self.assertEqual(
                        padded_batch[i][j][k], self.expected_padded_batch[i][j][k]
                    )

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import unittest

import torch
from pytext.data.utils import PAD_INDEX, UNK_INDEX
from pytext.models.embeddings.dict_embedding import DictEmbedding, PoolingType


class DictEmbeddingTest(unittest.TestCase):
    def test_basic(self):
        # Setup embedding
        num_embeddings = 5
        output_dim = 6
        embedding_module = DictEmbedding(
            num_embeddings=num_embeddings,
            embed_dim=output_dim,
            pooling_type=PoolingType.MEAN,
        )
        self.assertEqual(embedding_module.weight.size(0), num_embeddings)
        self.assertEqual(embedding_module.weight.size(1), output_dim)

        # The first and last tokens should be mapped to the zero vector.
        # This is due to the invariant that both unk and pad are considered
        # as padding indices.
        idx = torch.tensor(
            [UNK_INDEX, UNK_INDEX, 2, 3, 1, 1, 4, 1, PAD_INDEX, PAD_INDEX]
        ).unsqueeze(0)
        weights = torch.tensor(
            [0.3, 0.0, 0.8, 0.2, 0.0, 0.0, 1.0, 0.0, 0.3, 0.0], dtype=torch.float32
        ).unsqueeze(0)
        lens = torch.tensor([1, 2, 1, 1, 1]).unsqueeze(0)

        output = embedding_module(idx, weights, lens)

        self.assertAlmostEqual(output[0][0].sum().item(), 0)
        self.assertAlmostEqual(output[-1][-1].sum().item(), 0)

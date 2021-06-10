#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import unittest

import torch
from pytext.models.embeddings.int_weighted_multi_category_embedding import (
    IntWeightedMultiCategoryEmbedding,
)


class IntWeightedMultiCategoryEmbeddingTest(unittest.TestCase):
    def test_basic(self):
        # Setup embedding
        config = IntWeightedMultiCategoryEmbedding.Config(
            feature_buckets={1: 100, 2: 200}
        )
        embedding_module = IntWeightedMultiCategoryEmbedding.from_config(config)

        self.assertEqual(embedding_module.feature_embeddings["1"].weight.size(0), 100)
        self.assertEqual(
            embedding_module.feature_embeddings["1"].weight.size(1),
            config.embedding_dim,
        )
        self.assertEqual(embedding_module.feature_embeddings["2"].weight.size(0), 200)
        self.assertEqual(
            embedding_module.feature_embeddings["2"].weight.size(1),
            config.embedding_dim,
        )

        output = embedding_module(
            # feat ID -> (feat value - categories, offsets, weights)
            {
                1: (
                    torch.tensor([11, 12, 13], dtype=torch.int),
                    torch.tensor([0, 1], dtype=torch.int),
                    torch.tensor([0.0, 0.0, 0.0]),
                ),
                2: (
                    torch.tensor([], dtype=torch.int),
                    torch.tensor([0, 0], dtype=torch.int),
                    torch.tensor([]),
                ),
            }
        )

        self.assertEqual(list(output.size()), [2, 64])
        # First feature is embed to zeros as weight is zero. Second feature is embed to zero as it's empty.
        self.assertTrue(torch.all(output == torch.zeros([2, 64])))
        self.assertEqual(embedding_module.get_output_dim(), 64)

    def test_pooling(self):
        # Setup embedding
        config = IntWeightedMultiCategoryEmbedding.Config(
            pooling_type="max", feature_buckets={1: 100, 2: 200}
        )
        embedding_module = IntWeightedMultiCategoryEmbedding.from_config(config)

        output = embedding_module(
            {
                1: (
                    torch.tensor([11, 12, 13], dtype=torch.int),
                    torch.tensor([0, 1], dtype=torch.int),
                    torch.tensor([0.0, 0.0, 0.0]),
                ),
                2: (
                    torch.tensor([1], dtype=torch.int),
                    torch.tensor([0, 0], dtype=torch.int),
                    torch.tensor([1.0]),
                ),
            }
        )

        self.assertEqual(list(output.size()), [2, 32])
        self.assertEqual(embedding_module.get_output_dim(), 32)

    def test_pooling_mlp(self):
        # Setup embedding
        config = IntWeightedMultiCategoryEmbedding.Config(
            pooling_type="max",
            feature_buckets={1: 100, 2: 200},
            mlp_layer_dims=[40],
            ignore_weight=True,
            embedding_bag_mode="mean",
        )
        embedding_module = IntWeightedMultiCategoryEmbedding.from_config(config)

        self.assertEqual(embedding_module.get_output_dim(), 40)

        output = embedding_module(
            {
                1: (
                    torch.tensor([11, 12, 13], dtype=torch.int),
                    torch.tensor([0, 1], dtype=torch.int),
                    torch.tensor([0.0, 0.0, 0.0]),
                ),
                2: (
                    torch.tensor([1], dtype=torch.int),
                    torch.tensor([0, 0], dtype=torch.int),
                    torch.tensor([1.0]),
                ),
            }
        )

        self.assertEqual(list(output.size()), [2, 40])
        self.assertEqual(embedding_module.get_output_dim(), 40)

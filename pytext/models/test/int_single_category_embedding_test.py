#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import unittest

import torch
from pytext.models.embeddings.int_single_category_embedding import (
    IntSingleCategoryEmbedding,
)


class IntSingleCategoryEmbeddingTest(unittest.TestCase):
    def test_basic(self):
        # Setup embedding
        config = IntSingleCategoryEmbedding.Config(feature_buckets={1: 10, 2: 20})
        embedding_module = IntSingleCategoryEmbedding.from_config(config)

        self.assertEqual(embedding_module.feature_embeddings["1"].weight.size(0), 10)
        self.assertEqual(
            embedding_module.feature_embeddings["1"].weight.size(1),
            config.embedding_dim,
        )
        self.assertEqual(embedding_module.feature_embeddings["2"].weight.size(0), 20)
        self.assertEqual(
            embedding_module.feature_embeddings["2"].weight.size(1),
            config.embedding_dim,
        )

        output = embedding_module(
            {
                1: torch.tensor([11, 12, 13], dtype=torch.int),
                2: torch.tensor([21, 22, 23], dtype=torch.int),
                3: torch.tensor([31, 32, 33], dtype=torch.int),
            }
        )

        self.assertEqual(list(output.size()), [3, 64])
        self.assertEqual(embedding_module.get_output_dim(), 64)

    def test_pooling(self):
        # Setup embedding
        config = IntSingleCategoryEmbedding.Config(
            pooling_type="max", feature_buckets={1: 10, 2: 20}
        )
        embedding_module = IntSingleCategoryEmbedding.from_config(config)

        output = embedding_module(
            {
                1: torch.tensor([11, 12, 13], dtype=torch.int),
                2: torch.tensor([21, 22, 23], dtype=torch.int),
            }
        )

        self.assertEqual(list(output.size()), [3, 32])
        self.assertEqual(embedding_module.get_output_dim(), 32)

    def test_pooling_mlp(self):
        # Setup embedding
        config = IntSingleCategoryEmbedding.Config(
            pooling_type="max", feature_buckets={1: 10, 2: 20}, mlp_layer_dims=[40]
        )
        embedding_module = IntSingleCategoryEmbedding.from_config(config)

        output = embedding_module(
            {
                1: torch.tensor([11, 12, 13], dtype=torch.int),
                2: torch.tensor([21, 22, 23], dtype=torch.int),
            }
        )

        self.assertEqual(list(output.size()), [3, 40])
        self.assertEqual(embedding_module.get_output_dim(), 40)

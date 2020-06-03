#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import unittest

import torch
from pytext.models.embeddings.mlp_embedding import MLPEmbedding


class MLPEmbeddingTest(unittest.TestCase):
    def test_basic(self):
        # Setup embedding
        output_dim = 16
        embedding_module = MLPEmbedding(
            embedding_dim=4,
            embeddings_weight=None,
            init_range=[-1, 1],
            mlp_layer_dims=[output_dim],
        )
        self.assertEqual(embedding_module.embedding_dim, output_dim)

        # Check output shape
        input_batch_size, input_dim = 4, 4
        dense_features = torch.rand(size=[input_batch_size, input_dim])
        output_embedding = embedding_module(dense_features)
        expected_output_dims = [input_batch_size, output_dim]
        self.assertEqual(list(output_embedding.size()), expected_output_dims)

    def test_multi_mlp_layer_dims(self):
        output_dim = 16
        embedding_module = MLPEmbedding(
            embedding_dim=4,
            embeddings_weight=None,
            init_range=[-1, 1],
            mlp_layer_dims=[64, output_dim],
        )
        self.assertEqual(embedding_module.embedding_dim, output_dim)

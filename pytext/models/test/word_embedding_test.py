#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import unittest

import torch
from pytext.models.embeddings.word_embedding import WordEmbedding


class WordEmbeddingTest(unittest.TestCase):
    def test_basic(self):
        # Setup embedding
        num_embeddings = 5
        output_dim = 6
        embedding_module = WordEmbedding(
            num_embeddings=num_embeddings,
            embedding_dim=4,
            embeddings_weight=None,
            init_range=[-1, 1],
            unk_token_idx=4,
            mlp_layer_dims=[3, output_dim],
        )
        self.assertEqual(embedding_module.embedding_dim, output_dim)

        # Check output shape
        input_batch_size, input_len = 4, 6
        token_ids = torch.randint(
            low=0, high=num_embeddings, size=[input_batch_size, input_len]
        )
        output_embedding = embedding_module(token_ids)
        expected_output_dims = [input_batch_size, input_len, output_dim]
        self.assertEqual(list(output_embedding.size()), expected_output_dims)

    def test_none_mlp_layer_dims(self):
        num_embeddings = 5
        embedding_dim = 4
        embedding_module = WordEmbedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            embeddings_weight=None,
            init_range=[-1, 1],
            unk_token_idx=4,
            mlp_layer_dims=None,
        )
        self.assertEqual(embedding_module.embedding_dim, embedding_dim)

    def test_empty_mlp_layer_dims(self):
        num_embeddings = 5
        embedding_dim = 4
        embedding_module = WordEmbedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            embeddings_weight=None,
            init_range=[-1, 1],
            unk_token_idx=4,
            mlp_layer_dims=[],
        )
        self.assertEqual(embedding_module.embedding_dim, embedding_dim)

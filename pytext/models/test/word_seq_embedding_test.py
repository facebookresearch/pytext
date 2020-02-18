#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import unittest

import torch
from pytext.models.embeddings.word_seq_embedding import WordSeqEmbedding
from pytext.models.representations.bilstm import BiLSTM


class WordEmbeddingTest(unittest.TestCase):
    def test_basic(self):
        # Setup embedding
        num_embeddings = 5
        lstm_dim = 8
        embedding_module = WordSeqEmbedding(
            lstm_config=BiLSTM.Config(
                lstm_dim=lstm_dim, num_layers=2, bidirectional=True
            ),
            num_embeddings=num_embeddings,
            embedding_dim=4,
            embeddings_weight=None,
            init_range=[-1, 1],
            unk_token_idx=4,
        )
        # bidirectional
        output_dim = lstm_dim * 2
        self.assertEqual(embedding_module.embedding_dim, output_dim)

        # Check output shape
        input_batch_size, max_seq_len, max_token_count = 4, 3, 5
        token_seq_idx = torch.randint(
            low=0,
            high=num_embeddings,
            size=[input_batch_size, max_seq_len, max_token_count],
        )
        seq_token_count = torch.randint(
            low=1, high=max_token_count, size=[input_batch_size, max_seq_len]
        )
        seq_len = torch.randint(low=1, high=max_seq_len, size=[input_batch_size])
        output_embedding = embedding_module(token_seq_idx, seq_token_count, seq_len)

        expected_output_dims = [input_batch_size, max_seq_len, output_dim]
        self.assertEqual(list(output_embedding.size()), expected_output_dims)

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import unittest

from pytext.models.embeddings.char_embedding import CharacterEmbedding
from pytext.models.embeddings.embedding_list import EmbeddingList
from pytext.models.embeddings.word_embedding import WordEmbedding


class EmbeddingListTest(unittest.TestCase):
    def test_get_param_groups_for_optimizer(self):
        word_embedding = WordEmbedding(
            num_embeddings=5,
            embedding_dim=4,
            embeddings_weight=None,
            init_range=[-1, 1],
            unk_token_idx=4,
            mlp_layer_dims=[],
        )
        char_embedding = CharacterEmbedding(
            num_embeddings=5,
            embed_dim=4,
            out_channels=2,
            kernel_sizes=[1, 2],
            highway_layers=1,
            projection_dim=3,
        )
        embedding_list = EmbeddingList([word_embedding, char_embedding], concat=True)
        param_groups = embedding_list.get_param_groups_for_optimizer()
        self.assertEqual(len(param_groups), 1)

        param_names = set(param_groups[0].keys())
        expected_param_names = {name for name, _ in embedding_list.named_parameters()}
        self.assertSetEqual(param_names, expected_param_names)

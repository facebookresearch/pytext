#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

import torch
from pytext.config.module_config import PoolingType
from pytext.models.embeddings import (
    ContextualTokenEmbedding,
    DictEmbedding,
    EmbeddingList,
    WordEmbedding,
)
from pytext.models.semantic_parsers.rnng.jit import RNNGModel, RNNGParserJIT
from pytext.models.semantic_parsers.rnng.rnng_data_structures import CompositionalNN
from pytext.models.semantic_parsers.rnng.rnng_parser import RNNGParser


class RNNGJitTest(unittest.TestCase):
    def setUp(self):
        contextual_emb_dim = 1
        emb_module = EmbeddingList(
            embeddings=[
                WordEmbedding(num_embeddings=103, embedding_dim=100),
                DictEmbedding(
                    num_embeddings=59, embed_dim=10, pooling_type=PoolingType.MEAN
                ),
                ContextualTokenEmbedding(contextual_emb_dim),
            ],
            concat=True,
        )
        self.parser = RNNGParserJIT(
            ablation=RNNGParser.Config.AblationParams(),
            constraints=RNNGParser.Config.RNNGConstraints(),
            lstm_num_layers=2,
            lstm_dim=32,
            max_open_NT=10,
            dropout=0.4,
            num_actions=20,
            shift_idx=0,
            reduce_idx=1,
            ignore_subNTs_roots=[8, 15],
            valid_NT_idxs=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            + [12, 13, 14, 15, 16, 17, 18, 19],
            valid_IN_idxs=[2, 4, 7, 8, 10, 12, 13, 14, 15],
            valid_SL_idxs=[3, 5, 6, 9, 11, 16, 17, 18, 19],
            embedding=RNNGModel.trace_embedding(emb_module, contextual_emb_dim),
            embedding_dim=emb_module.embedding_dim,
            p_compositional=CompositionalNN(lstm_dim=32, device="cpu"),
            device="cpu",
        )
        self.parser.train()

    def test_forward(self):
        tokens = torch.tensor([[96, 34, 11, 19, 15, 12]])
        seq_lens = torch.tensor([tokens.shape[1]])
        dict_feat = (
            torch.tensor([[1, 1, 1, 1, 3, 2, 1, 1, 4, 1, 7, 1]]),
            torch.tensor(
                [[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]]
            ),
            torch.tensor([[1, 1, 2, 1, 1, 1]]),
        )
        contextual_feat = torch.tensor([[0.2, 0.2, 0.3, 0.2, 0.2, 0.3]])
        target_actions = [[2, 0, 0, 0, 0, 5, 0, 0, 1, 1]]
        actions, scores = self.parser(
            tokens=tokens,
            seq_lens=seq_lens,
            dict_feat=dict_feat,
            actions=target_actions,
            contextual_token_embeddings=contextual_feat,
        )[0]
        self.assertGreater(actions.shape[1], tokens.shape[1])
        self.assertEqual(actions.shape[0:2], scores.shape[0:2])
        self.assertEqual(scores.shape[2], 20)

    def test_save(self):
        self.parser.save("/tmp/test.pt1")

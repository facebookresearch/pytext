#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import unittest

import torch.nn as nn
from pytext.contrib.pytext_lib import models
from pytext.contrib.pytext_lib.transforms import VocabTransform


class TestModels(unittest.TestCase):
    def setUp(self):
        self.base_dir = os.path.join(os.path.dirname(__file__), "data")

    def test_load_roberta(self):
        model = models.RobertaModel(
            model_path=None,
            dense_dim=0,
            embedding_dim=32,
            out_dim=2,
            vocab_size=105,
            num_attention_heads=1,
            num_encoder_layers=1,
            output_dropout=0.4,
        )
        self.assertTrue(isinstance(model, nn.Module))

    def test_load_doc_model(self):
        transform = VocabTransform(os.path.join(self.base_dir, "vocab_dummy"))
        vocab = transform.vocab
        model = models.DocModel(
            pretrained_embeddings_path=os.path.join(
                self.base_dir, "word_embedding_dummy"
            ),
            embedding_dim=300,
            mlp_layer_dims=(2,),
            skip_header=True,
            kernel_num=1,
            kernel_sizes=(3, 4, 5),
            decoder_hidden_dims=(2,),
            vocab=vocab,
        )
        self.assertTrue(isinstance(model, nn.Module))

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

    def test_load_xlmr_dummy(self):
        model = models.xlmr_dummy_binary_doc_classifier(pretrained=False)
        assert isinstance(model, nn.Module)

    def test_load_roberta_base(self):
        model = models.roberta_base_binary_doc_classifier(pretrained=False)
        assert isinstance(model, nn.Module)

    def test_load_xlmr_base(self):
        model = models.xlmr_base_binary_doc_classifier(pretrained=False)
        assert isinstance(model, nn.Module)

    def test_load_doc_model(self):
        transform = VocabTransform(os.path.join(self.base_dir, "vocab_dummy"))
        vocab = transform.vocab
        model = models.DocClassificationModel(
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
        assert isinstance(model, nn.Module)

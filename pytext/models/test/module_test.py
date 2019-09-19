#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

from pytext.config.component import create_model
from pytext.config.field_config import DictFeatConfig, FeatureConfig, WordFeatConfig
from pytext.data import CommonMetadata
from pytext.fields import FieldMeta
from pytext.models.doc_model import DocModel


class VocabStub:
    def __init__(self):
        self.itos = []
        self.stoi = {}


def mock_metadata():
    meta = CommonMetadata
    field_meta = FieldMeta()
    field_meta.vocab = VocabStub()
    field_meta.vocab_size = 10
    field_meta.pretrained_embeds_weight = None
    field_meta.unk_token_idx = 0
    meta.features = {"word_feat": field_meta, "dict_feat": field_meta}
    meta.target = field_meta
    return meta


class ModuleTest(unittest.TestCase):
    # TODO () Port this test to DocModel
    def DISABLED_test_freeze_word_embedding(self):
        model = create_model(
            DocModel.Config(),
            FeatureConfig(
                word_feat=WordFeatConfig(freeze=True, mlp_layer_dims=[4]),
                dict_feat=DictFeatConfig(),
            ),
            metadata=mock_metadata(),
        )
        # word embedding
        for param in model.embedding[0].word_embedding.parameters():
            self.assertFalse(param.requires_grad)
        for param in model.embedding[0].mlp.parameters():
            self.assertTrue(param.requires_grad)

        # dict feat embedding
        for param in model.embedding[1].parameters():
            self.assertTrue(param.requires_grad)

    # TODO () Port this test to DocModel
    def DISABLED_test_freeze_all_embedding(self):
        model = create_model(
            DocModel.Config(), FeatureConfig(freeze=True), metadata=mock_metadata()
        )
        for param in model.embedding.parameters():
            self.assertFalse(param.requires_grad)

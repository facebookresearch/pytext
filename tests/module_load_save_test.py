#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import itertools
import os
import tempfile
import unittest

from pytext.common.constants import DatasetFieldName
from pytext.config.component import create_model
from pytext.config.field_config import FeatureConfig
from pytext.data import CommonMetadata
from pytext.fields import FieldMeta
from pytext.models.decoders.mlp_decoder import MLPDecoder
from pytext.models.doc_model import DocModel_Deprecated
from pytext.models.representations.bilstm_doc_attention import BiLSTMDocAttention


class VocabStub:
    def __init__(self):
        self.itos = []
        self.stoi = {}


class ModuleLoadSaveTest(unittest.TestCase):
    def setUp(self):
        self.embedding_file, self.embedding_path = tempfile.mkstemp()
        self.decoder_file, self.decoder_path = tempfile.mkstemp()
        self.representation_file, self.representation_path = tempfile.mkstemp()

    def tearDown(self):
        for f in (self.embedding_file, self.decoder_file, self.representation_file):
            os.close(f)
        for p in (self.embedding_path, self.decoder_path, self.representation_path):
            os.remove(p)

    def test_load_save(self):
        text_field_meta = FieldMeta()
        text_field_meta.vocab = VocabStub()
        text_field_meta.vocab_size = 4
        text_field_meta.unk_token_idx = 1
        text_field_meta.pad_token_idx = 0
        text_field_meta.pretrained_embeds_weight = None
        label_meta = FieldMeta()
        label_meta.vocab = VocabStub()
        label_meta.vocab_size = 3
        metadata = CommonMetadata()
        metadata.features = {DatasetFieldName.TEXT_FIELD: text_field_meta}
        metadata.target = label_meta

        saved_model = create_model(
            DocModel_Deprecated.Config(
                representation=BiLSTMDocAttention.Config(
                    save_path=self.representation_path
                ),
                decoder=MLPDecoder.Config(save_path=self.decoder_path),
            ),
            FeatureConfig(save_path=self.embedding_path),
            metadata,
        )
        saved_model.save_modules()

        loaded_model = create_model(
            DocModel_Deprecated.Config(
                representation=BiLSTMDocAttention.Config(
                    load_path=self.representation_path
                ),
                decoder=MLPDecoder.Config(load_path=self.decoder_path),
            ),
            FeatureConfig(load_path=self.embedding_path),
            metadata,
        )

        random_model = create_model(
            DocModel_Deprecated.Config(
                representation=BiLSTMDocAttention.Config(), decoder=MLPDecoder.Config()
            ),
            FeatureConfig(),
            metadata,
        )

        # Loaded and saved modules should be equal. Neither should be equal to
        # a randomly initialised model.

        for p1, p2, p3 in itertools.zip_longest(
            saved_model.embedding.parameters(),
            loaded_model.embedding.parameters(),
            random_model.embedding.parameters(),
        ):
            self.assertTrue(p1.equal(p2))
            self.assertFalse(p3.equal(p1))
            self.assertFalse(p3.equal(p2))

        for p1, p2, p3 in itertools.zip_longest(
            saved_model.representation.parameters(),
            loaded_model.representation.parameters(),
            random_model.representation.parameters(),
        ):
            self.assertTrue(p1.equal(p2))
            self.assertFalse(p3.equal(p1))
            self.assertFalse(p3.equal(p2))

        for p1, p2, p3 in itertools.zip_longest(
            saved_model.decoder.parameters(),
            loaded_model.decoder.parameters(),
            random_model.decoder.parameters(),
        ):
            self.assertTrue(p1.equal(p2))
            self.assertFalse(p3.equal(p1))
            self.assertFalse(p3.equal(p2))

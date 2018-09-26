#!/usr/bin/env python3

import itertools
import os
import tempfile
import unittest

from pytext.common.constants import DatasetFieldName
from pytext.config.component import create_model
from pytext.data import CommonMetadata
from pytext.fields import FieldMeta
from pytext.models.decoders.mlp_decoder import MLPDecoder
from pytext.models.doc_model import DocModel
from pytext.models.embeddings.token_embedding import FeatureConfig
from pytext.models.representations.bilstm_pooling import BiLSTMPooling


class ModuleLoadSaveTest(unittest.TestCase):
    def setUp(self):
        self.decoder_file, self.decoder_path = tempfile.mkstemp()
        self.representation_file, self.representation_path = tempfile.mkstemp()

    def tearDown(self):
        os.close(self.decoder_file)
        os.close(self.representation_file)
        os.remove(self.decoder_path)
        os.remove(self.representation_path)

    def test_load_save(self):
        text_field_meta = FieldMeta()
        text_field_meta.vocab_size = 1
        text_field_meta.unk_token_idx = 0
        text_field_meta.pad_token_idx = 0
        label_meta = FieldMeta()
        label_meta.vocab_size = 1
        metadata = CommonMetadata()
        metadata.features = {DatasetFieldName.TEXT_FIELD: text_field_meta}
        metadata.labels = {DatasetFieldName.DOC_LABEL_FIELD: label_meta}

        saved_model = create_model(
            DocModel.Config(
                representation=BiLSTMPooling.Config(save_path=self.representation_path),
                decoder=MLPDecoder.Config(save_path=self.decoder_path),
            ),
            FeatureConfig(),
            metadata,
        )
        saved_model.save_modules()

        loaded_model = create_model(
            DocModel.Config(
                representation=BiLSTMPooling.Config(load_path=self.representation_path),
                decoder=MLPDecoder.Config(load_path=self.decoder_path),
            ),
            FeatureConfig(),
            metadata,
        )

        random_model = create_model(
            DocModel.Config(
                representation=BiLSTMPooling.Config(), decoder=MLPDecoder.Config()
            ),
            FeatureConfig(),
            metadata,
        )

        # Loaded and saved modules should be equal. Neither should be equal to
        # a randomly initialised model.
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

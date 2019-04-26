#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

from pytext.common.constants import DatasetFieldName
from pytext.config.doc_classification import ModelInput, ModelInputConfig, TargetConfig
from pytext.config.field_config import FeatureConfig
from pytext.data import DocClassificationDataHandler, RawData
from pytext.data.featurizer import SimpleFeaturizer
from pytext.utils.test import import_tests_module


tests_module = import_tests_module()


class DocClassificationDataHandlerTest(unittest.TestCase):
    def setUp(self):
        handler_config = DocClassificationDataHandler.Config()
        handler_config.columns_to_read.append(ModelInput.DENSE_FEAT)
        self.data_handler = DocClassificationDataHandler.from_config(
            DocClassificationDataHandler.Config(),
            ModelInputConfig(),
            TargetConfig(),
            featurizer=SimpleFeaturizer.from_config(
                SimpleFeaturizer.Config(), FeatureConfig()
            ),
        )

    def test_create_from_config(self):
        expected_columns = [
            RawData.DOC_LABEL,
            RawData.TEXT,
            RawData.DICT_FEAT,
            ModelInput.DENSE_FEAT,
        ]
        # check that the list of columns is as expected
        self.assertTrue(self.data_handler.raw_columns == expected_columns)

    def test_read_from_file(self):
        file_name = tests_module.test_file("train_dense_features_tiny.tsv")
        data = list(
            self.data_handler.read_from_file(file_name, self.data_handler.raw_columns)
        )

        # Check if the data has 10 rows and 6 columns
        self.assertEqual(len(data), 10)
        self.assertEqual(len(data[0]), 4)

        self.assertEqual(data[0][RawData.DOC_LABEL], "alarm/modify_alarm")

    def test_tokenization(self):
        file_name = tests_module.test_file("train_dense_features_tiny.tsv")

        data = list(
            self.data_handler.read_from_file(file_name, self.data_handler.raw_columns)
        )
        data = list(self.data_handler.preprocess(data))

        # test tokenization without language-specific tokenizers
        self.assertEqual(
            data[0][ModelInput.WORD_FEAT][0], "16:24:datetime,39:57:datetime"
        )
        self.assertIsNotNone(data[0][ModelInput.DENSE_FEAT])
        self.assertEqual(data[0][DatasetFieldName.NUM_TOKENS], 1)

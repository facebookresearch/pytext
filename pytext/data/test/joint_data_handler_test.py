#!/usr/bin/env python3

import unittest

from pytext.common.constants import DFColumn, DatasetFieldName
from pytext.config.field_config import FeatureConfig, LabelConfig
from pytext.data import JointModelDataHandler
from pytext.data.featurizer import SimpleFeaturizer


class JointDataHandlerTest(unittest.TestCase):
    def setUp(self):
        self.data_handler = JointModelDataHandler.from_config(
            JointModelDataHandler.Config(),
            FeatureConfig(),
            LabelConfig(),
            featurizer=SimpleFeaturizer.from_config(
                SimpleFeaturizer.Config(), FeatureConfig()),
        )

    def test_create_from_config(self):
        expected_columns = [
            DFColumn.DOC_LABEL,
            DFColumn.WORD_LABEL,
            DFColumn.UTTERANCE,
            DFColumn.DICT_FEAT,
            DFColumn.DOC_WEIGHT,
            DFColumn.WORD_WEIGHT,
        ]

        # check that the list of columns is as expected
        self.assertTrue(self.data_handler.raw_columns == expected_columns)

    def test_read_from_file(self):
        file_name = "pytext/tests/data/train_data_tiny.tsv"
        data = self.data_handler.read_from_file(
            file_name, self.data_handler.raw_columns)

        # Check if the data has 10 rows and 6 columns
        self.assertEqual(len(data), 10)
        self.assertEqual(len(data[0]), 6)

        self.assertEqual(
            data[0][DFColumn.UTTERANCE],
            "change my alarm tomorrow to wake me up 30 minutes earlier"
        )

    def test_tokenization(self):
        file_name = "pytext/tests/data/train_data_tiny.tsv"

        data = self.data_handler.read_from_file(
            file_name, self.data_handler.raw_columns)
        data = list(self.data_handler.preprocess(data))

        # test tokenization without language-specific tokenizers
        self.assertEqual(data[0][DatasetFieldName.TEXT_FIELD][0], "change")
        self.assertEqual(data[4][DatasetFieldName.TEXT_FIELD][2], "alarm")

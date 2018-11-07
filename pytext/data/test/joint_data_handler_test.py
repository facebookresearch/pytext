#!/usr/bin/env python3

import unittest

from pytext.common.constants import DFColumn, DatasetFieldName
from pytext.config.field_config import FeatureConfig, LabelConfig
from pytext.data import JointModelDataHandler
from pytext.data.featurizer import SimpleFeaturizer


class JointDataHandlerTest(unittest.TestCase):
    def test_create_from_config(self):
        data_handler = JointModelDataHandler.from_config(
            JointModelDataHandler.Config(),
            FeatureConfig(),
            LabelConfig(),
            featurizer=SimpleFeaturizer(),
        )
        expected_columns = [
            DFColumn.DOC_LABEL,
            DFColumn.WORD_LABEL,
            DFColumn.UTTERANCE,
            DFColumn.DICT_FEAT,
            DFColumn.DOC_WEIGHT,
            DFColumn.WORD_WEIGHT,
        ]

        # check that the list of columns is as expected
        self.assertTrue(data_handler.raw_columns == expected_columns)

    def test_read_from_file(self):
        file_name = "pytext/tests/data/music_train_tiny.tsv"
        data_handler = JointModelDataHandler.from_config(
            JointModelDataHandler.Config(),
            FeatureConfig(),
            LabelConfig(),
            featurizer=SimpleFeaturizer(),
        )

        data = data_handler.read_from_file(file_name, data_handler.raw_columns)

        # Check if the data has 10 rows and 6 columns
        self.assertEqual(len(data), 10)
        self.assertEqual(len(data[0]), 6)

        self.assertEqual(
            data[0][DFColumn.UTTERANCE], "Do i have any of Jeremiah's albums"
        )

    def test_tokenization(self):
        file_name = "pytext/tests/data/music_train_tiny.tsv"
        data_handler = JointModelDataHandler.from_config(
            JointModelDataHandler.Config(),
            FeatureConfig(),
            LabelConfig(),
            featurizer=SimpleFeaturizer(),
        )

        data = data_handler.read_from_file(file_name, data_handler.raw_columns)
        data = list(data_handler.preprocess(data))

        # test tokenization without language-specific tokenizers
        self.assertEqual(data[0][DatasetFieldName.TEXT_FIELD][0], "do")
        self.assertEqual(data[4][DatasetFieldName.TEXT_FIELD][2], "song")

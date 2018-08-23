#!/usr/bin/env python3

import unittest

from pytext.common.constants import DFColumn
from pytext.config.field_config import FeatureConfig, LabelConfig
from pytext.data.joint_data_handler import JointModelDataHandler


class JointDataHandlerTest(unittest.TestCase):
    def test_create_from_config(self):
        data_handler = JointModelDataHandler.from_config(
            JointModelDataHandler.Config(), FeatureConfig(), LabelConfig()
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
            JointModelDataHandler.Config(), FeatureConfig(), LabelConfig()
        )

        df = data_handler.read_from_file(file_name, data_handler.raw_columns)

        # Check if the df has 10 rows and 6 columns
        self.assertEqual(len(df), 10)
        self.assertEqual(len(list(df)), 6)

        self.assertEqual(
            df[DFColumn.UTTERANCE][0], "Do i have any of Jeremiah's albums"
        )

    def test_tokenization(self):
        file_name = "pytext/tests/data/music_train_tiny.tsv"
        data_handler = JointModelDataHandler.from_config(
            JointModelDataHandler.Config(), FeatureConfig(), LabelConfig()
        )

        df = data_handler.read_from_file(file_name, data_handler.raw_columns)
        df = data_handler._preprocess_df(df)

        # test tokenization without language-specific tokenizers
        self.assertEqual(df[DFColumn.MODEL_FEATS][0].tokens[0], "do")
        self.assertEqual(df[DFColumn.MODEL_FEATS][4].tokens[2], "song")

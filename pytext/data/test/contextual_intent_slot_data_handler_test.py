#!/usr/bin/env python3

import unittest

from pytext.common.constants import DFColumn
from pytext.config.field_config import FeatureConfig, LabelConfig
from pytext.data import ContextualIntentSlotModelDataHandler
from pytext.data.featurizer import SimpleFeaturizer


class ContextualIntentSlotModelDataHandlerTest(unittest.TestCase):
    def setUp(self):
        file_name = "pytext/tests/data/contextual_intent_slot_train_tiny.tsv"
        self.dh = ContextualIntentSlotModelDataHandler.from_config(
            ContextualIntentSlotModelDataHandler.Config(),
            FeatureConfig(),
            LabelConfig(),
            featurizer=SimpleFeaturizer(),
        )

        self.data = self.dh.read_from_file(file_name, self.dh.raw_columns)

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
        self.assertTrue(self.dh.raw_columns == expected_columns)

    def test_read_from_file(self):
        # Check if the data has 10 rows and 6 columns
        self.assertEqual(len(self.data), 10)
        self.assertEqual(len(self.data[0]), 6)
        self.assertEqual(self.data[0][DFColumn.UTTERANCE], '["Hey", "Youd love this"]')

    def test_intermediate_result(self):
        data = self.dh.gen_dataset(self.data)
        self.assertListEqual(data.examples[0].word_feat, ["youd", "love", "this"])
        self.assertListEqual(
            data.examples[0].seq_word_feat, [["hey"], ["youd", "love", "this"]]
        )
        self.assertEqual(data.examples[0].doc_weight, "0.2")
        self.assertEqual(data.examples[0].word_weight, "0.5")
        self.assertEqual(data.examples[0].raw_word_label, "")
        self.assertListEqual(data.examples[0].token_range, [0, 4, 5, 9, 10, 14])
        self.assertEqual(data.examples[0].utterance, '["Hey", "Youd love this"]')

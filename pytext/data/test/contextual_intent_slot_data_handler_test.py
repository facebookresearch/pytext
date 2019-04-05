#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

from pytext.common.constants import DFColumn
from pytext.config.contextual_intent_slot import ModelInput, ModelInputConfig
from pytext.config.field_config import DocLabelConfig, WordLabelConfig
from pytext.data import ContextualIntentSlotModelDataHandler
from pytext.data.featurizer import SimpleFeaturizer
from pytext.utils.test import import_tests_module


tests_module = import_tests_module()


class ContextualIntentSlotModelDataHandlerTest(unittest.TestCase):
    def setUp(self):
        file_name = tests_module.test_file("contextual_intent_slot_train_tiny.tsv")
        self.dh = ContextualIntentSlotModelDataHandler.from_config(
            ContextualIntentSlotModelDataHandler.Config(),
            ModelInputConfig(),
            [DocLabelConfig(), WordLabelConfig()],
            featurizer=SimpleFeaturizer(SimpleFeaturizer.Config(), ModelInputConfig()),
        )

        self.data = list(self.dh.read_from_file(file_name, self.dh.raw_columns))

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
        self.assertListEqual(data.examples[0].token_range, [(0, 4), (5, 9), (10, 14)])
        self.assertEqual(data.examples[0].utterance, '["Hey", "Youd love this"]')


class ContextualIntentSlotModelDataHandlerDenseTest(unittest.TestCase):
    def test_read_file_with_dense_features(self):
        data_handler_config = ContextualIntentSlotModelDataHandler.Config()
        data_handler_config.columns_to_read.append(ModelInput.DENSE)
        dense_file_name = tests_module.test_file(
            "contextual_intent_slot_train_tiny_dense.tsv"
        )
        data_handler = ContextualIntentSlotModelDataHandler.from_config(
            data_handler_config,
            ModelInputConfig(),
            [DocLabelConfig(), WordLabelConfig()],
            featurizer=SimpleFeaturizer(SimpleFeaturizer.Config(), ModelInputConfig()),
        )

        dense_data = list(
            data_handler.read_from_file(dense_file_name, data_handler.raw_columns)
        )
        self.assertEqual(dense_data[0][ModelInput.DENSE], "[0,1,2,3,4]")

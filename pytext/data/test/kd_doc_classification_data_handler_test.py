#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

from pytext.common.constants import DFColumn
from pytext.config.doc_classification import ModelInput, ModelInputConfig, TargetConfig
from pytext.config.field_config import FeatureConfig, Target
from pytext.data import DocClassificationDataHandler
from pytext.data.featurizer import SimpleFeaturizer
from pytext.utils.test import import_tests_module


tests_module = import_tests_module()


class KDDocClassificationDataHandlerTest(unittest.TestCase):
    def setUp(self):
        file_name = tests_module.test_file("knowledge_distillation_test_tiny.tsv")
        label_config_dict = {"target_prob": True}
        data_handler_dict = {
            "columns_to_read": [
                "text",
                "target_probs",
                "target_logits",
                "target_labels",
                "doc_label",
            ]
        }
        self.data_handler = DocClassificationDataHandler.from_config(
            DocClassificationDataHandler.Config(**data_handler_dict),
            ModelInputConfig(),
            TargetConfig(**label_config_dict),
            featurizer=SimpleFeaturizer.from_config(
                SimpleFeaturizer.Config(), FeatureConfig()
            ),
        )
        self.data = list(
            self.data_handler.read_from_file(file_name, self.data_handler.raw_columns)
        )

    def test_create_from_config(self):
        expected_columns = [
            DFColumn.UTTERANCE,
            DFColumn.TARGET_PROBS,
            DFColumn.TARGET_LOGITS,
            DFColumn.TARGET_LABELS,
            DFColumn.DOC_LABEL,
        ]
        # check that the list of columns is as expected
        self.assertTrue(self.data_handler.raw_columns == expected_columns)

    def test_read_from_file(self):
        # Check if the data has 10 rows and 5 columns
        self.assertEqual(len(self.data), 10)
        self.assertEqual(len(self.data[0]), 5)

        self.assertEqual(self.data[0][DFColumn.UTTERANCE], "Who R U ?")
        self.assertEqual(
            self.data[0][DFColumn.TARGET_PROBS],
            "[-0.005602254066616297, -5.430975914001465]",
        )
        self.assertEqual(
            self.data[0][DFColumn.TARGET_LABELS], '["cu:other", "cu:ask_Location"]'
        )

    def test_tokenization(self):
        data = list(self.data_handler.preprocess(self.data))

        # test tokenization without language-specific tokenizers
        self.assertListEqual(data[0][ModelInput.WORD_FEAT], ["who", "r", "u", "?"])
        self.assertListEqual(
            data[0][Target.TARGET_PROB_FIELD],
            [-0.005602254066616297, -5.430975914001465],
        )

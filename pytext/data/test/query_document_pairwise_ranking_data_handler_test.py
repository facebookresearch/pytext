#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

from pytext.config.field_config import FeatureConfig
from pytext.config.query_document_pairwise_ranking import ModelInput, ModelInputConfig
from pytext.data import QueryDocumentPairwiseRankingDataHandler
from pytext.data.featurizer import SimpleFeaturizer
from pytext.utils.test import import_tests_module


tests_module = import_tests_module()


class QueryDocumentPairwiseRankingDataHandlerTest(unittest.TestCase):
    def setUp(self):
        simple_featurizer_config = SimpleFeaturizer.Config()
        simple_featurizer_config.split_regex = r""
        simple_featurizer_config.convert_to_bytes = True

        self.data_handler = QueryDocumentPairwiseRankingDataHandler.from_config(
            QueryDocumentPairwiseRankingDataHandler.Config(),
            ModelInputConfig(),
            [],
            featurizer=SimpleFeaturizer.from_config(
                simple_featurizer_config, FeatureConfig()
            ),
        )

    def test_create_from_config(self):
        expected_columns = [
            ModelInput.QUERY,
            ModelInput.POS_RESPONSE,
            ModelInput.NEG_RESPONSE,
        ]
        # check that the list of columns is as expected
        print(self.data_handler.raw_columns)
        self.assertTrue(self.data_handler.raw_columns == expected_columns)

    def test_read_from_file(self):
        file_name = tests_module.test_file("query_document_pairwise_ranking_tiny.tsv")
        data = list(
            self.data_handler.read_from_file(file_name, self.data_handler.raw_columns)
        )
        # Check if the data has 4 rows and 3 columns
        self.assertEqual(len(data), 4)
        self.assertEqual(len(data[0]), 3)
        self.assertEqual(data[1][ModelInput.QUERY], "query")
        self.assertEqual(data[1][ModelInput.POS_RESPONSE], "response1")
        self.assertEqual(data[1][ModelInput.NEG_RESPONSE], "response2")

    def test_tokenization(self):
        file_name = tests_module.test_file("query_document_pairwise_ranking_tiny.tsv")
        data = list(
            self.data_handler.read_from_file(file_name, self.data_handler.raw_columns)
        )
        data = list(self.data_handler.preprocess(data))
        print(data[0][ModelInput.QUERY])
        # test tokenization: must split into chars (ascii string input)
        self.assertEqual(data[0][ModelInput.QUERY], ["q", "u", "e", "r", "y"])

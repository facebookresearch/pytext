#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

import numpy as np
from pytext.config.field_config import FeatureConfig, WordLabelConfig
from pytext.data import BPTTLanguageModelDataHandler
from pytext.data.featurizer import SimpleFeaturizer
from pytext.utils.test_utils import import_tests_module


tests_module = import_tests_module()
FILE_NAME = tests_module.test_file("alarm_lm_tiny.tsv")
BATCH_SIZE = 4


class BPTTLanguageModelDataHandlerTest(unittest.TestCase):
    def test_data_handler(self):
        data_handler = BPTTLanguageModelDataHandler.from_config(
            BPTTLanguageModelDataHandler.Config(bptt_len=4),
            FeatureConfig(),
            WordLabelConfig(),
            featurizer=SimpleFeaturizer.from_config(
                SimpleFeaturizer.Config(), FeatureConfig()
            ),
        )
        data_handler.init_metadata_from_path(FILE_NAME, FILE_NAME, FILE_NAME)

        train_iter = data_handler.get_train_iter_from_path(FILE_NAME, BATCH_SIZE)

        batches = [t for t in train_iter]
        # There are two batches in the tiny dataset
        self.assertEqual(len(batches), 2)

        # batches of tuple(input, target, context)
        # input -> tuple(input_sequences, sequence_length)
        # input_sequence -> tensor of dim (bsize, max_seq_length)
        np.testing.assert_array_equal(
            batches[0][0][0],
            [[15, 19, 12, 16], [3, 13, 21, 8], [20, 7, 23, 4], [6, 5, 7, 22]],
        )
        # sequence_length -> tensor of dim (bsize)
        np.testing.assert_array_equal(batches[0][0][1], [4, 4, 4, 4])

        # target -> tensor of same dim as input_sequences (bsize, max_seq_length)
        np.testing.assert_array_equal(
            batches[0][1][0],
            [[19, 12, 16, 14], [13, 21, 8, 3], [7, 23, 4, 3], [5, 7, 22, 10]],
        )

        np.testing.assert_array_equal(
            batches[1][0][0], [[14, 17, 11], [3, 5, 18], [3, 8, 4], [10, 4, 9]]
        )
        np.testing.assert_array_equal(batches[1][0][1], [3, 3, 3, 3])
        np.testing.assert_array_equal(
            batches[1][1][0], [[17, 11, 4], [5, 18, 6], [8, 4, 3], [4, 9, 1]]
        )

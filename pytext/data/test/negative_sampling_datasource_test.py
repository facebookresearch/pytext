#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

from pytext.data.sources.data_source import SafeFileWrapper
from pytext.data.sources.negative_sampling_data_source import NegativeSamplingDataSource
from pytext.data.sources.tsv import BlockShardedTSV
from pytext.utils.test import import_tests_module


tests_module = import_tests_module()


class NegativeSamplingDataSourceTest(unittest.TestCase):
    def setUp(self):
        fname = tests_module.test_file("train_dense_features_tiny.tsv")
        train1 = BlockShardedTSV(
            SafeFileWrapper(fname),
            field_names=["label1", "slots1", "text1", "dense1"],
            block_id=0,
            num_blocks=2,
        )
        train2 = BlockShardedTSV(
            SafeFileWrapper(fname),
            field_names=["label2", "slots2", "text2", "dense2"],
            block_id=1,
            num_blocks=2,
        )
        schema = {"text1": str, "label1": str, "text2": str, "label2": str}
        self.data = NegativeSamplingDataSource(
            schema, train1, train2, None, None, None, None
        )

    def test_read_data_source(self):
        data = list(self.data.train)
        print(data)
        self.assertEqual(4, len(data))
        example = next(iter(data))
        self.assertEqual(4, len(example))
        self.assertEqual({"label1", "text1", "label2", "text2"}, set(example))

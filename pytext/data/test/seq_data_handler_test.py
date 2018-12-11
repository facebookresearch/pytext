#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

import numpy as np
from pytext.common.constants import DFColumn
from pytext.config.field_config import DocLabelConfig, FeatureConfig
from pytext.data import SeqModelDataHandler
from pytext.data.featurizer import SimpleFeaturizer


class SeqModelDataHandlerTest(unittest.TestCase):
    def setUp(self):
        self.train_data = [
            {
                DFColumn.DOC_LABEL: "cu:discuss_where",
                DFColumn.UTTERANCE: '["where do you wanna meet?", "MPK"]',
            }
        ]

        self.eval_data = [
            {
                DFColumn.DOC_LABEL: "cu:discuss_where",
                DFColumn.UTTERANCE: '["how about SF?", "sounds good"]',
            },
            {DFColumn.DOC_LABEL: "cu:other", DFColumn.UTTERANCE: '["lol"]'},
        ]

        self.test_data = [
            {
                DFColumn.DOC_LABEL: "cu:discuss_where",
                DFColumn.UTTERANCE: '["MPK sounds good to me"]',
            },
            {
                DFColumn.DOC_LABEL: "cu:other",
                DFColumn.UTTERANCE: '["great", "awesome"]',
            },
        ]

        self.dh = SeqModelDataHandler.from_config(
            SeqModelDataHandler.Config(),
            FeatureConfig(),
            DocLabelConfig(),
            featurizer=SimpleFeaturizer.from_config(
                SimpleFeaturizer.Config(), FeatureConfig()
            ),
        )

    def test_intermediate_result(self):
        data = self.dh.gen_dataset(self.train_data)
        self.assertListEqual(
            data.examples[0].word_feat,
            [["where", "do", "you", "wanna", "meet?"], ["mpk"]],
        )

    def test_process_data(self):
        self.dh.init_metadata_from_raw_data(
            self.train_data, self.eval_data, self.test_data
        )
        train_iter = self.dh.get_train_iter_from_raw_data(self.train_data, 1)
        for input, target, _ in train_iter:
            np.testing.assert_array_almost_equal(
                input[0][0].numpy(), [[6, 2, 7, 5, 3], [4, 1, 1, 1, 1]]
            )
            np.testing.assert_array_almost_equal(input[1].numpy(), [2])
            np.testing.assert_array_almost_equal(target[0].numpy(), [0])

#!/usr/bin/env python3

import unittest
import pandas as pd
import numpy as np

from pytext.common.constants import DFColumn
from pytext.config.field_config import FeatureConfig, LabelConfig, DocLabelConfig
from pytext.data.seq_data_handler import SeqModelDataHandler


class SeqModelDataHandlerTest(unittest.TestCase):
    def setUp(self):
        self.train_data = pd.DataFrame(
            {
                DFColumn.DOC_LABEL: ["cu:discuss_where"],
                DFColumn.UTTERANCE: [
                    '["where do you wanna meet?", "MPK"]'
                ],
            }
        )

        self.eval_data = pd.DataFrame(
            {
                DFColumn.DOC_LABEL: ["cu:discuss_where", "cu:other"],
                DFColumn.UTTERANCE: [
                    '["how about SF?", "sounds good"]',
                    '["lol"]',
                ],
            }
        )

        self.test_data = pd.DataFrame(
            {
                DFColumn.DOC_LABEL: ["cu:discuss_where", "cu:other"],
                DFColumn.UTTERANCE: [
                    '["MPK sounds good to me"]',
                    '["great", "awesome"]',
                ],
            }
        )

        self.dh = SeqModelDataHandler.from_config(
            SeqModelDataHandler.Config(),
            FeatureConfig(),
            LabelConfig(doc_label=DocLabelConfig()),
        )

    def test_intermediate_result(self):
        data = self.dh.gen_dataset(self.train_data)
        self.assertListEqual(
            data.examples[0].text,
            [["where", "do", "you", "wanna", "meet", "?"], ["mpk"]],
        )

    def test_process_data(self):
        self.dh.init_metadata_from_df(self.train_data, self.eval_data, self.test_data)
        train_iter = self.dh.get_train_batch_from_df((self.train_data,), (1,))[0]
        for input, target, _ in train_iter:
            np.testing.assert_array_almost_equal(
                input[0][0].numpy(),
                [[7, 3, 8, 6, 4, 2], [5, 1, 1, 1, 1, 1]],
            )
            np.testing.assert_array_almost_equal(
                input[1].numpy(), [2],
            )
            np.testing.assert_array_almost_equal(
                target[0].numpy(), [0],
            )

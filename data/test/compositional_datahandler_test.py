#!/usr/bin/env python3

import unittest
import pandas as pd
import numpy as np
from pytext.data.compositional_data_handler import CompositionalDataHandler
from pytext.common.constants import DFColumn


class CompositionalDataHandlerTest(unittest.TestCase):
    def setUp(self):
        self.train_data = pd.DataFrame(
            {
                DFColumn.DOC_LABEL: ["IN:GET_EVENT"],
                DFColumn.WORD_LABEL: [
                    [
                        {
                            "id": "SL:DATE_TIME",
                            "span": {"start": 21, "end": 28},
                            "text": "tonight",
                        }
                    ]
                ],
                DFColumn.UTTERANCE: ["What events can I go tonight"],
                DFColumn.DICT_FEAT: [""],
            }
        )

        self.eval_data = pd.DataFrame(
            {
                DFColumn.DOC_LABEL: ["IN:GET_EVENT"],
                DFColumn.WORD_LABEL: [
                    [
                        {
                            "id": "SL:ATTRIBUTE_EVENT",
                            "span": {"start": 14, "end": 19},
                            "text": "adult",
                        },
                        {
                            "id": "SL:DATE_TIME",
                            "span": {"start": 27, "end": 39},
                            "text": "this weekend",
                        },
                    ]
                ],
                DFColumn.UTTERANCE: ["Are there any adult events this weekend"],
                DFColumn.DICT_FEAT: [""],
            }
        )

        self.test_data = pd.DataFrame(
            {
                DFColumn.DOC_LABEL: ["IN:GET_INFO_ROAD_CONDITION"],
                DFColumn.WORD_LABEL: [
                    [
                        {
                            "id": "SL:ROAD_CONDITION",
                            "span": {"start": 9, "end": 21},
                            "text": "any flooding",
                        },
                        {
                            "id": "SL:DESTINATION",
                            "span": {"start": 36, "end": 41},
                            "text": "Karen",
                            "subframe": {
                                "utterance": "Karen",
                                "domain": "",
                                "intent": "IN:GET_LOCATION_HOME",
                                "slots": [
                                    {
                                        "id": "SL:CONTACT",
                                        "span": {"start": 0, "end": 5},
                                        "text": "Karen",
                                    }
                                ],
                                "span": {"start": 0, "end": 5},
                            },
                        },
                    ]
                ],
                DFColumn.UTTERANCE: ["Is there any flooding on the way to Karen's?"],
                DFColumn.DICT_FEAT: [""],
            }
        )

        columns = [
            DFColumn.DOC_LABEL,
            DFColumn.WORD_LABEL,
            DFColumn.UTTERANCE,
            DFColumn.DICT_FEAT,
        ]

        self.dh = CompositionalDataHandler(raw_columns=columns)

    def test_intermediate_result(self):
        data = self.dh.gen_dataset(self.train_data)
        actions_expected = [
            "IN:GET_EVENT",
            "SHIFT",
            "SHIFT",
            "SHIFT",
            "SHIFT",
            "SHIFT",
            "SL:DATE_TIME",
            "SHIFT",
            "REDUCE",
            "REDUCE",
        ]
        self.assertListEqual(
            data.examples[0].text, ["what", "events", "can", "i", "go", "tonight"]
        )
        self.assertListEqual(data.examples[0].action_idx_feature, actions_expected)
        self.assertListEqual(data.examples[0].action_idx_label, actions_expected)

    def test_process_data(self):
        self.dh.init_metadata_from_df(self.train_data, self.eval_data, self.test_data)
        print(self.dh.metadata['feature_vocabs']['text'].itos)
        train_iter = self.dh.get_train_batch_from_df((self.train_data,), (1,))[0]
        for input, target, _ in train_iter:
            print(input)
            np.testing.assert_array_almost_equal(
                input[0][0].numpy(), [[6, 4, 5, 2, 3, 7]]
            )
            np.testing.assert_array_almost_equal(
                input[2].numpy(), [[2, 2, 1, 4, 1, 1, 1, 1, 1, 3]]
            )
            np.testing.assert_array_almost_equal(
                target[0].numpy(), [[2, 2, 1, 4, 1, 1, 1, 1, 1, 3]]
            )

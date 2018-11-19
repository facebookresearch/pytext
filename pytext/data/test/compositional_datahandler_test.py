#!/usr/bin/env python3
import unittest

import numpy as np
from pytext.common.constants import DFColumn
from pytext.config.field_config import FeatureConfig, WordFeatConfig
from pytext.data import CompositionalDataHandler
from pytext.data.featurizer import SimpleFeaturizer


class CompositionalDataHandlerTest(unittest.TestCase):
    def setUp(self):
        self.train_data = [
            {
                DFColumn.DOC_LABEL: "IN:GET_EVENT",
                DFColumn.WORD_LABEL: [
                    {
                        "id": "SL:DATE_TIME",
                        "span": {"start": 21, "end": 28},
                        "text": "tonight",
                    }
                ],
                DFColumn.UTTERANCE: "What events can I go tonight",
                DFColumn.DICT_FEAT: "",
                DFColumn.SEQLOGICAL: "[IN:GET_EVENT What events can I go [SL:DATE_TIME tonight ] ]",
            }
        ]

        self.eval_data = [
            {
                DFColumn.DOC_LABEL: "IN:GET_EVENT",
                DFColumn.WORD_LABEL: [
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
                ],
                DFColumn.UTTERANCE: "Are there any adult events this weekend",
                DFColumn.DICT_FEAT: "",
                DFColumn.SEQLOGICAL: "[IN:GET_EVENT Are there any [SL:ATTRIBUTE_EVENT adult ] events [SL:DATE_TIME this weekend ] ]",
            }
        ]

        self.test_data = [
            {
                DFColumn.DOC_LABEL: "IN:GET_INFO_ROAD_CONDITION",
                DFColumn.WORD_LABEL: [
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
                ],
                DFColumn.UTTERANCE: "Is there any flooding on the way to Karen's?",
                DFColumn.DICT_FEAT: "",
                DFColumn.SEQLOGICAL: "[IN:GET_INFO_ROAD_CONDITION Is there [SL:ROAD_CONDITION any flooding ] on the way to [SL:DESTINATION [IN:GET_LOCATION_HOME [SL:CONTACT Karen 's ? ] ] ] ]",
            }
        ]

        self.dh = CompositionalDataHandler.from_config(
            CompositionalDataHandler.Config(min_item_freq_for_vocab=1),
            FeatureConfig(word_feat=WordFeatConfig(vocab_from_all_data=True)),
            featurizer=SimpleFeaturizer.from_config(
                SimpleFeaturizer.Config(lowercase_tokens=False), FeatureConfig()
            ),
        )

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
            data.examples[0].word_feat, ["what", "events", "can", "i", "go", "tonight"]
        )
        self.assertListEqual(data.examples[0].action_idx_feature, actions_expected)
        self.assertListEqual(data.examples[0].action_idx_label, actions_expected)

    def test_train_tensors(self):
        self.dh.init_metadata_from_raw_data(
            self.train_data, self.eval_data, self.test_data
        )
        print(self.dh.features["action_idx_feature"].vocab.stoi)
        for input, _, _ in self.dh.get_train_iter_from_raw_data(
            self.train_data, batch_size=1
        ):
            print(input)
            # input = [token_batch, seq_lens_batch, dict_feat_batch, actions_batch]
            np.testing.assert_array_almost_equal(
                input[0].numpy(), [[12, 1, 5, 7, 6, 10]]
            )  # tokens
            np.testing.assert_array_almost_equal(
                input[1].numpy(), [6]
            )  # sequence length
            self.assertTrue(input[2] is None)  # no dict feats
            np.testing.assert_array_almost_equal(
                input[3], [[2, 0, 0, 0, 0, 0, 3, 0, 1, 1]]
            )  # actions

    def test_test_tensors(self):
        self.dh.init_metadata_from_raw_data(
            self.train_data, self.eval_data, self.test_data
        )
        for input, target, _ in self.dh.get_test_iter_from_raw_data(
            self.eval_data, batch_size=1
        ):
            # input = [token_batch, seq_lens_batch, dict_feat_batch]
            np.testing.assert_array_almost_equal(
                input[0].numpy(), [[4, 8, 3, 2, 1, 9, 11]]
            )  # tokens
            np.testing.assert_array_almost_equal(
                input[1].numpy(), [7]
            )  # sequence length
            self.assertTrue(input[2] is None)  # no dict feats
            np.testing.assert_array_almost_equal(
                target.numpy(), [[2, 0, 0, 0, 4, 0, 1, 0, 3, 0, 0, 1, 1]]
            )  # actions target

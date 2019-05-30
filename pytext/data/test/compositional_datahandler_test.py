#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
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
                        "span": {"start": 24, "end": 34},
                        "text": "8 pm today",
                    }
                ],
                DFColumn.UTTERANCE: "What EVENTS can I go to at 8 pm today",
                DFColumn.DICT_FEAT: "",
                DFColumn.SEQLOGICAL: "[IN:GET_EVENT What EVENTS can I go to at [SL:DATE_TIME 8 pm today ] ]",
            }
        ]

        self.eval_data = [
            {
                DFColumn.DOC_LABEL: "IN:GET_EVENT",
                DFColumn.WORD_LABEL: [
                    {
                        "id": "SL:ATTRIBUTE_EVENT",
                        "span": {"start": 14, "end": 17},
                        "text": "fun",
                    },
                    {
                        "id": "SL:DATE_TIME",
                        "span": {"start": 25, "end": 37},
                        "text": "this weekend",
                    },
                ],
                DFColumn.UTTERANCE: "Are there any fun events this weekend",
                DFColumn.DICT_FEAT: "",
                DFColumn.SEQLOGICAL: "[IN:GET_EVENT Are there any [SL:ATTRIBUTE_EVENT fun ] events [SL:DATE_TIME this weekend ] ]",
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
            CompositionalDataHandler.Config(),
            FeatureConfig(
                word_feat=WordFeatConfig(vocab_from_all_data=True, min_freq=1)
            ),
            featurizer=SimpleFeaturizer.from_config(
                SimpleFeaturizer.Config(lowercase_tokens=True), FeatureConfig()
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
            "SHIFT",
            "SHIFT",
            "SL:DATE_TIME",
            "SHIFT",
            "SHIFT",
            "SHIFT",
            "REDUCE",
            "REDUCE",
        ]
        self.assertListEqual(
            data.examples[0].word_feat,
            ["what", "events", "can", "i", "go", "to", "at", "8", "pm", "today"],
        )
        self.assertListEqual(data.examples[0].action_idx_feature, actions_expected)
        self.assertListEqual(data.examples[0].action_idx_label, actions_expected)

    def test_train_tensors(self):
        self.dh.init_metadata_from_raw_data(
            self.train_data, self.eval_data, self.test_data
        )
        self.assertSetEqual(
            set(self.dh.features["word_feat"].vocab.stoi),
            {
                "<unk>",
                "<unk>-NUM",
                "what",
                "events",
                "can",
                "i",
                "go",
                "to",
                "at",
                "pm",
                "today",
                "are",
                "there",
                "any",
                "fun",
                "this",
                "weekend",
            },
        )
        print(self.dh.features["word_feat"].vocab.stoi)
        print(self.dh.features["action_idx_feature"].vocab.stoi)
        for input, _, _ in self.dh.get_train_iter_from_raw_data(
            self.train_data, batch_size=1
        ):
            print(input)
            # input = [token_batch, seq_lens_batch, dict_feat_batch, actions_batch]
            np.testing.assert_array_almost_equal(
                input[0].numpy(), [[16, 2, 6, 9, 8, 13, 5, 1, 10, 14]]
            )  # tokens
            np.testing.assert_array_almost_equal(
                input[1].numpy(), [10]
            )  # sequence length
            self.assertTrue(input[2] is None)  # no dict feats
            np.testing.assert_array_almost_equal(
                input[3], [[2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 1]]
            )  # actions

    def test_test_tensors(self):
        self.dh.init_metadata_from_raw_data(
            self.train_data, self.eval_data, self.test_data
        )
        print(self.dh.features["word_feat"].vocab.stoi)
        print(self.dh.features["action_idx_feature"].vocab.stoi)
        for input, target, _ in self.dh.get_test_iter_from_raw_data(
            self.eval_data, batch_size=1
        ):
            # input = [token_batch, seq_lens_batch, dict_feat_batch]
            np.testing.assert_array_almost_equal(
                input[0].numpy(), [[4, 11, 3, 7, 2, 12, 15]]
            )  # tokens
            np.testing.assert_array_almost_equal(
                input[1].numpy(), [7]
            )  # sequence length
            self.assertTrue(input[2] is None)  # no dict feats
            np.testing.assert_array_almost_equal(
                target.numpy(), [[2, 0, 0, 0, 4, 0, 1, 0, 3, 0, 0, 1, 1]]
            )  # actions target

    def test_min_freq(self):
        """
        Test that UNKification is triggered when min_freq is 2.
        """
        custom_dh = CompositionalDataHandler.from_config(
            CompositionalDataHandler.Config(),
            FeatureConfig(
                word_feat=WordFeatConfig(vocab_from_all_data=True, min_freq=2)
            ),
            featurizer=SimpleFeaturizer.from_config(
                SimpleFeaturizer.Config(lowercase_tokens=True), FeatureConfig()
            ),
        )
        custom_dh.init_metadata_from_raw_data(
            self.train_data, self.eval_data, self.test_data
        )
        # <unk>-NUM = <unk> for numeric tokens
        self.assertSetEqual(
            set(custom_dh.features["word_feat"].vocab.stoi),
            {"<unk>", "<unk>-NUM", "<unk>", "<unk>", "events"},
        )

    def test_uppercase_tokens(self):
        """
        Test that the text is not lower-cased when lowercase_tokens is False.
        """
        custom_dh = CompositionalDataHandler.from_config(
            CompositionalDataHandler.Config(),
            FeatureConfig(
                word_feat=WordFeatConfig(vocab_from_all_data=True, min_freq=1)
            ),
            featurizer=SimpleFeaturizer.from_config(
                SimpleFeaturizer.Config(lowercase_tokens=False), FeatureConfig()
            ),
        )
        custom_dh.init_metadata_from_raw_data(
            self.train_data, self.eval_data, self.test_data
        )
        self.assertSetEqual(
            set(custom_dh.features["word_feat"].vocab.stoi),
            {
                "<unk>",
                "<unk>-NUM",
                "What",
                "EVENTS",
                "can",
                "I",
                "go",
                "to",
                "at",
                "pm",
                "today",
                "Are",
                "there",
                "any",
                "fun",
                "events",
                "this",
                "weekend",
            },
        )

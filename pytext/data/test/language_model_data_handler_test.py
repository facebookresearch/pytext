#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import unittest
from typing import Dict

import numpy as np
from pytext.common.constants import BatchContext, DatasetFieldName, DFColumn, VocabMeta
from pytext.config.component import create_featurizer
from pytext.config.field_config import FeatureConfig, WordLabelConfig
from pytext.data import LanguageModelDataHandler
from pytext.data.featurizer import SimpleFeaturizer
from pytext.fields import Field, TextFeatureField
from pytext.utils.test import import_tests_module


tests_module = import_tests_module()
FILE_NAME = tests_module.test_file("alarm_lm_tiny.tsv")
BATCH_SIZE = 5


class LanguageModelDataHandlerTest(unittest.TestCase):
    @classmethod
    def create_language_model_data_handler(cls) -> LanguageModelDataHandler:
        # TODO: Refactor this after Shicong refactors PyText config and removes
        # Thrift. After that directly use Data Handler's from config method
        # with synthetic configs
        columns = [DFColumn.UTTERANCE]
        features: Dict[str, Field] = {
            DatasetFieldName.TEXT_FIELD: TextFeatureField(
                eos_token=VocabMeta.EOS_TOKEN, init_token=VocabMeta.INIT_TOKEN
            )
        }

        return LanguageModelDataHandler(
            raw_columns=columns,
            features=features,
            labels={},
            featurizer=create_featurizer(SimpleFeaturizer.Config(), FeatureConfig()),
        )

    def test_data_handler(self):
        data_handler = self._init_data_handler()
        text_feat_meta = data_handler.metadata.features[DatasetFieldName.TEXT_FIELD]
        self.assertEqual(text_feat_meta.vocab_size, 25)
        self.assertEqual(text_feat_meta.pad_token_idx, 1)
        self.assertEqual(text_feat_meta.unk_token_idx, 0)
        self.assertEqual(text_feat_meta.init_token_idx, 2)
        self.assertEqual(text_feat_meta.eos_token_idx, 3)

        train_iter = data_handler.get_train_iter_from_path(FILE_NAME, BATCH_SIZE)

        batches = [t for t in train_iter]
        # There is only one batch in the tiny dataset
        self.assertEqual(len(batches), 1)

        # batch -> tuple(input, target, context)
        batch = batches[0]

        # input -> tuple(input_sequences, sequence_length)
        # input_sequence -> tensor of dim (bsize, max_seq_length)
        np.testing.assert_array_equal(
            batch[0][0],
            [
                [2, 16, 20, 13, 17, 15, 18, 12, 5, 4, 14, 22],
                [2, 9, 4, 6, 19, 7, 21, 8, 3, 1, 1, 1],
                [2, 9, 5, 4, 7, 6, 8, 3, 1, 1, 1, 1],
                [2, 23, 11, 5, 10, 3, 1, 1, 1, 1, 1, 1],
                [2, 24, 5, 4, 3, 1, 1, 1, 1, 1, 1, 1],
            ],
        )
        # sequence_length -> tensor of dim (bsize)
        np.testing.assert_array_equal(batch[0][1], [12, 9, 8, 6, 5])

        # target -> tensor of same dim as input_sequences (bsize, max_seq_length)
        np.testing.assert_array_equal(
            batch[1],
            [
                [16, 20, 13, 17, 15, 18, 12, 5, 4, 14, 22, 3],
                [9, 4, 6, 19, 7, 21, 8, 3, 1, 1, 1, 1],
                [9, 5, 4, 7, 6, 8, 3, 1, 1, 1, 1, 1],
                [23, 11, 5, 10, 3, 1, 1, 1, 1, 1, 1, 1],
                [24, 5, 4, 3, 1, 1, 1, 1, 1, 1, 1, 1],
            ],
        )

    def test_sharding(self):
        data_handler = self._init_data_handler()
        num_shards = 2
        batch_size = 5
        train_iter_1 = data_handler.get_train_iter_from_path(
            FILE_NAME, batch_size, 0, num_shards
        )
        train_iter_2 = data_handler.get_train_iter_from_path(
            FILE_NAME, batch_size, 1, num_shards
        )
        batches_1 = list(train_iter_1)
        batches_2 = list(train_iter_2)
        self.assertEqual(len(batches_1), len(batches_2))

        shard_size_1 = len(train_iter_1.batches.dataset)
        shard_size_2 = len(train_iter_1.batches.dataset)
        self.assertEqual(shard_size_1, shard_size_2)

        # Shard 1 & 2 iterator shouldn't have dummy batches
        CONTEXT_INDEX = 2
        for b in batches_1 + batches_2:
            assert BatchContext.IGNORE_LOSS not in b[CONTEXT_INDEX]

        # shard #1 took shard with input data index [1, 2, 3]
        # shard #2 took shard with input data index [3, 4, 5]
        # we pad shard #2 to make every shard the same size
        test_batch = batches_1[0]
        # first batch in shard #1 is row # 2 and 3 reordered by sort_key
        np.testing.assert_array_equal(
            test_batch[1], [[9, 4, 6, 19, 7, 21, 8, 3], [24, 5, 4, 3, 1, 1, 1, 1]]
        )

        test_batch = batches_2[0]
        # second batch in shard #2 is row # 3 and 4 reordered by sort_key
        np.testing.assert_array_equal(
            test_batch[1], [[23, 11, 5, 10, 3], [24, 5, 4, 3, 1]]
        )

    def _init_data_handler(self):
        data_handler = LanguageModelDataHandler.from_config(
            LanguageModelDataHandler.Config(),
            FeatureConfig(),
            WordLabelConfig(),
            featurizer=create_featurizer(SimpleFeaturizer.Config(), FeatureConfig()),
            shuffle=False,
        )
        data_handler.init_metadata_from_path(FILE_NAME, FILE_NAME, FILE_NAME)
        return data_handler

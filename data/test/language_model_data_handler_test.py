#!/usr/bin/env python3

import unittest
from typing import Dict

import numpy as np
from pytext.common.constants import (
    DatasetFieldName,
    DFColumn,
    VocabMeta,
)
from pytext.data.language_model_data_handler import LanguageModelDataHandler
from pytext.data.shared_featurizer import SharedFeaturizer
from pytext.fields import Field, TextFeatureField


FILE_NAME = "pytext/tests/data/alarm_lm_tiny.tsv"
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
            featurizer=SharedFeaturizer(),
            raw_columns=columns,
            features=features,
            labels={},
        )

    def test_data_handler(self):
        data_handler = self.create_language_model_data_handler()
        data_handler.init_metadata_from_path(FILE_NAME, FILE_NAME, FILE_NAME)
        text_feat_meta = data_handler.metadata.features[DatasetFieldName.TEXT_FIELD]
        self.assertEqual(text_feat_meta.vocab_size, 25)
        self.assertEqual(text_feat_meta.pad_token_idx, 1)
        self.assertEqual(text_feat_meta.unk_token_idx, 0)
        self.assertEqual(text_feat_meta.init_token_idx, 2)
        self.assertEqual(text_feat_meta.eos_token_idx, 3)

        train_iter = data_handler.get_train_batch_from_path(
            (FILE_NAME,), (BATCH_SIZE,)
        )[0]

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

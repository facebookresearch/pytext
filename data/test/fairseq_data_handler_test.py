#!/usr/bin/env python3

import unittest

from pytext.common.constants import DatasetFieldName, DFColumn
from pytext.config.field_config import FeatureConfig, LabelConfig
from pytext.data import FairSeqDataHandler
from pytext.data.featurizer import SimpleFeaturizer


class FairSeqDataHandlerTest(unittest.TestCase):
    def test_create_from_config(self):
        data_handler = FairSeqDataHandler.from_config(
            FairSeqDataHandler.Config(target_featurizer=SimpleFeaturizer.Config()),
            FeatureConfig(),
            LabelConfig(),
            featurizer=SimpleFeaturizer(),
        )
        expected_columns = [DFColumn.SOURCE_SEQUENCE, DFColumn.TARGET_SEQUENCE]
        # check that the list of columns is as expected
        self.assertTrue(data_handler.raw_columns == expected_columns)

    def test_read_from_path(self):
        file_name = "pytext/tests/data/compositional_seq2seq_unit.tsv"
        data_handler = FairSeqDataHandler.from_config(
            FairSeqDataHandler.Config(target_featurizer=SimpleFeaturizer.Config()),
            FeatureConfig(),
            LabelConfig(),
            featurizer=SimpleFeaturizer(),
        )

        df = data_handler.read_from_file(file_name, data_handler.raw_columns)

        # Check if the df has 10 rows and 2 columns
        self.assertEqual(df.shape, (10, 2))
        self.assertEqual(df[DFColumn.SOURCE_SEQUENCE][0], "delays in tempe")
        self.assertEqual(
            df[DFColumn.TARGET_SEQUENCE][0].strip(),
            "[IN:GET_INFO_TRAFFIC delays in [SL:LOCATION tempe ] ]",
        )

    def test_batching(self):
        file_name = "pytext/tests/data/compositional_seq2seq_unit.tsv"
        data_handler = FairSeqDataHandler.from_config(
            FairSeqDataHandler.Config(target_featurizer=SimpleFeaturizer.Config()),
            FeatureConfig(),
            LabelConfig(),
            featurizer=SimpleFeaturizer(),
        )
        data_handler.init_metadata_from_path(file_name, file_name, file_name)
        train_iter = data_handler.get_train_iter_from_path(file_name, 2)
        for inputs, _, _ in train_iter:
            # Prev output token starts with eos
            self.assertEqual(
                inputs[-1][0][0],
                data_handler.labels[DatasetFieldName.TARGET_SEQ_FIELD]
                .get_meta()
                .eos_token_idx,
            )

#!/usr/bin/env python3

import unittest
from pytext.data import DataHandler
from pytext.common.constants import DFColumn


class DataHandlerTest(unittest.TestCase):
    def test_read_from_file(self):
        file_name = "pytext/tests/data/music_train_tiny.tsv"
        columns = [
            DFColumn.DOC_LABEL,
            DFColumn.WORD_LABEL,
            DFColumn.UTTERANCE,
            DFColumn.DICT_FEAT,
        ]

        df = DataHandler.read_from_file(file_name, columns)
        for col in columns:
            self.assertTrue(col in df, "{} must in the df data".format(col))

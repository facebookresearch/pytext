#!/usr/bin/env python3

import unittest

from pytext.common.constants import DFColumn
from pytext.data.data_handler import DataHandler


class DataHandlerTest(unittest.TestCase):
    def test_read_from_csv(self):
        file_name = "pytext/tests/data/music_train_tiny.tsv"
        columns = [
            DFColumn.DOC_LABEL,
            DFColumn.WORD_LABEL,
            DFColumn.UTTERANCE,
            DFColumn.DICT_FEAT,
        ]

        data = DataHandler.read_from_file(file_name, columns)
        for col in columns:
            self.assertTrue(col in data[0], "{} must in the data".format(col))
        self.assertEqual(
            "aloha_assistant:music/isAvailable", data[0][DFColumn.DOC_LABEL]
        )
        self.assertEqual(
            "17:25:music/artistName,28:34:music/type", data[0][DFColumn.WORD_LABEL]
        )
        self.assertEqual(
            "Do i have any of Jeremiah's albums", data[0][DFColumn.UTTERANCE]
        )
        self.assertEqual(
            '{"tokenFeatList": [{"tokenIdx": 3, "features": {"texasHandler_companies": 1.0}}]}',
            data[0][DFColumn.DICT_FEAT],
        )

    def test_read_partially_from_csv(self):
        file_name = "pytext/tests/data/music_train_tiny.tsv"
        columns = {DFColumn.DOC_LABEL: 0, DFColumn.UTTERANCE: 2}

        data = DataHandler.read_from_file(file_name, columns)
        for col in columns:
            self.assertTrue(col in data[0], "{} must in the data".format(col))
        self.assertEqual(
            "aloha_assistant:music/isAvailable", data[0][DFColumn.DOC_LABEL]
        )
        self.assertEqual(
            "Do i have any of Jeremiah's albums", data[0][DFColumn.UTTERANCE]
        )


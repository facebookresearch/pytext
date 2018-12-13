#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

from pytext.common.constants import DFColumn
from pytext.data.data_handler import DataHandler
from pytext.utils.test_utils import import_tests_module


tests_module = import_tests_module()


class DataHandlerTest(unittest.TestCase):
    def test_read_from_csv(self):
        file_name = tests_module.test_file("train_data_tiny.tsv")
        columns = [
            DFColumn.DOC_LABEL,
            DFColumn.WORD_LABEL,
            DFColumn.UTTERANCE,
            DFColumn.DICT_FEAT,
        ]

        data = DataHandler.read_from_file(file_name, columns)
        for col in columns:
            self.assertTrue(col in data[0], "{} must in the data".format(col))
        self.assertEqual("alarm/modify_alarm", data[0][DFColumn.DOC_LABEL])
        self.assertEqual("16:24:datetime,39:57:datetime", data[0][DFColumn.WORD_LABEL])
        self.assertEqual(
            "change my alarm tomorrow to wake me up 30 minutes earlier",
            data[0][DFColumn.UTTERANCE],
        )
        self.assertEqual("", data[0][DFColumn.DICT_FEAT])

    def test_read_partially_from_csv(self):
        file_name = tests_module.test_file("train_data_tiny.tsv")
        columns = {DFColumn.DOC_LABEL: 0, DFColumn.UTTERANCE: 2}

        data = DataHandler.read_from_file(file_name, columns)
        for col in columns:
            self.assertTrue(col in data[0], "{} must in the data".format(col))
        self.assertEqual("alarm/modify_alarm", data[0][DFColumn.DOC_LABEL])
        self.assertEqual(
            "change my alarm tomorrow to wake me up 30 minutes earlier",
            data[0][DFColumn.UTTERANCE],
        )

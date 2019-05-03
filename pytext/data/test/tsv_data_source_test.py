#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest
from typing import List

from pytext.data.sources.data_source import SafeFileWrapper
from pytext.data.sources.tsv import SessionTSVDataSource, TSVDataSource
from pytext.utils.test import import_tests_module


tests_module = import_tests_module()


class TSVDataSourceTest(unittest.TestCase):
    def setUp(self):
        self.data = TSVDataSource(
            SafeFileWrapper(tests_module.test_file("train_dense_features_tiny.tsv")),
            SafeFileWrapper(tests_module.test_file("test_dense_features_tiny.tsv")),
            eval_file=None,
            field_names=["label", "slots", "text", "dense"],
            schema={"text": str, "label": str},
        )

    def test_read_data_source(self):
        data = list(self.data.train)
        self.assertEqual(10, len(data))
        example = next(iter(data))
        self.assertEqual(2, len(example))
        self.assertEqual({"label", "text"}, set(example))

    def test_quoting(self):
        data_source = TSVDataSource(
            SafeFileWrapper(tests_module.test_file("test_tsv_quoting.tsv")),
            SafeFileWrapper(tests_module.test_file("test_tsv_quoting.tsv")),
            eval_file=None,
            field_names=["label", "text"],
            schema={"text": str, "label": str},
        )

        data = list(data_source.train)
        self.assertEqual(4, len(data))

    def test_read_test_data_source(self):
        data = list(self.data.test)
        self.assertEqual(5, len(data))
        example = next(iter(data))
        self.assertEqual(2, len(example))
        self.assertEqual({"label", "text"}, set(example))

    def test_read_eval_data_source(self):
        data = list(self.data.eval)
        self.assertEqual(0, len(data))

    def test_iterate_training_data_multiple_times(self):
        train = self.data.train
        data = list(train)
        data2 = list(train)
        self.assertEqual(10, len(data))
        self.assertEqual(10, len(data2))
        example = next(iter(data2))
        self.assertEqual(2, len(example))
        self.assertEqual({"label", "text"}, set(example))

    def test_read_data_source_with_column_remapping(self):
        data_source = TSVDataSource(
            SafeFileWrapper(tests_module.test_file("train_dense_features_tiny.tsv")),
            SafeFileWrapper(tests_module.test_file("test_dense_features_tiny.tsv")),
            eval_file=None,
            field_names=["remapped_label", "slots", "remapped_text", "dense"],
            column_mapping={"remapped_label": "label", "remapped_text": "text"},
            schema={"text": str, "label": str},
        )

        data = list(data_source.train)
        self.assertEqual(10, len(data))
        example = next(iter(data))
        self.assertEqual(2, len(example))
        self.assertEqual({"label", "text"}, set(example))

    def test_read_data_source_with_utf8_issues(self):
        schema = {"text": str, "label": str}
        data_source = TSVDataSource.from_config(
            TSVDataSource.Config(
                train_filename=tests_module.test_file("test_utf8_errors.tsv"),
                field_names=["label", "text"],
            ),
            schema,
        )
        list(data_source.train)


class SessionTSVDataSourceTest(unittest.TestCase):
    def setUp(self):
        self.data = SessionTSVDataSource(
            SafeFileWrapper(tests_module.test_file("seq_tagging_example.tsv")),
            field_names=["session_id", "intent", "goals", "label"],
            schema={"intent": List[str], "goals": List[str], "label": List[str]},
        )

    def test_read_session_data(self):
        self.assertEqual(3, len(list(self.data.train)))
        # validate multiple iteration
        self.assertEqual(3, len(list(self.data.train)))
        it = iter(self.data.train)
        example = next(it)
        self.assertEqual(4, len(example))
        self.assertEqual("id1", example["session_id"])
        self.assertEqual(["int11", "int12"], example["intent"])
        self.assertEqual(["g11", "g12"], example["goals"])
        self.assertEqual(["0", "0"], example["label"])
        example = next(it)
        example = next(it)
        self.assertEqual("id3", example["session_id"])
        self.assertEqual(["int31", "int32", "int33"], example["intent"])
        self.assertEqual(["g31", "g32", "g33"], example["goals"])
        self.assertEqual(["0", "1", "1"], example["label"])

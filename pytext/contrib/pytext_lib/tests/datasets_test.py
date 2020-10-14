#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import csv
import os
import unittest

from pytext.contrib.pytext_lib.data.datasets import JsonlDataset, TsvDataset


class TestDatasets(unittest.TestCase):
    def setUp(self):
        pwd = os.path.dirname(__file__)
        self.tsv_path = os.path.join(pwd, "data/dev_10.tsv")
        self.jsonl_path = os.path.join(pwd, "data/dev_10.jsonl")

        self.text, self.labels = [], []
        with open(self.tsv_path, "rt", newline="") as fin:
            reader = csv.DictReader(fin, delimiter="\t", fieldnames=["text", "label"])
            for row in reader:
                self.text.append(row["text"])
                self.labels.append(row["label"])

    def check_dataset_line_by_line(self, dataset):
        # Ensure the dataset reports the correct size.
        self.assertEqual(len(dataset), len(self.text))

        # Ensure the dataset returns the correct data, in the correct order.
        for i in range(len(dataset)):
            self.assertEqual(dataset[i]["text"], self.text[i])
            self.assertEqual(str(dataset[i]["label"]), str(self.labels[i]))

    def test_tsv_dataset(self):
        # test at various page sizes
        for page_size in [1, 7, 10, 11]:
            # test explicitly setting dataset size vs inferring it
            for total_size in [None, 10]:
                dataset = TsvDataset(
                    self.tsv_path,
                    ["text", "label"],
                    page_size=page_size,
                    total_size=total_size,
                )

                self.check_dataset_line_by_line(dataset)
                # test a second time to ensure the iterator is reinitialized correctly
                self.check_dataset_line_by_line(dataset)

    def test_jsonl_dataset(self):
        # test at various page sizes
        for page_size in [1, 7, 10, 11]:
            # test explicitly setting dataset size vs inferring it
            for total_size in [None, 10]:
                dataset = JsonlDataset(
                    self.jsonl_path, page_size=page_size, total_size=total_size
                )

                self.check_dataset_line_by_line(dataset)
                # test a second time to ensure the iterator is reinitialized correctly
                self.check_dataset_line_by_line(dataset)

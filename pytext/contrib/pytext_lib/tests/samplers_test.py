#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import random
import unittest

from pytext.contrib.pytext_lib.data.datasets.samplers import (
    PooledBatchSampler,
    SortedSampler,
)
from torch.utils.data import DataLoader


class TestSamplers(unittest.TestCase):
    def setUp(self):
        self.data = ["a", "bb", "ccc", "dddd", "e" * 5, "f" * 6, "g" * 7]
        random.shuffle(self.data)

    def test_sorted_sampler(self):
        descending_sampler = SortedSampler(self.data, key=lambda x: -len(x))
        data_loader = DataLoader(self.data, sampler=descending_sampler)
        assert list(data_loader) == [
            ["g" * 7],
            ["f" * 6],
            ["e" * 5],
            ["dddd"],
            ["ccc"],
            ["bb"],
            ["a"],
        ]

        ascending_sampler = SortedSampler(self.data, key=len)
        data_loader = DataLoader(self.data, sampler=ascending_sampler)
        assert list(data_loader) == [
            ["a"],
            ["bb"],
            ["ccc"],
            ["dddd"],
            ["e" * 5],
            ["f" * 6],
            ["g" * 7],
        ]

    def test_pooled_batch_sampler(self):
        unsorted_sampler = PooledBatchSampler(self.data, batch_size=2, drop_last=False)
        batch_lens = [len(batch) for batch in unsorted_sampler]
        assert batch_lens.count(2) == 3 and batch_lens.count(1) == 1

        descending_sampler = PooledBatchSampler(
            self.data, batch_size=3, drop_last=False, key=lambda x: -len(x)
        )
        data_loader = DataLoader(self.data, batch_sampler=descending_sampler)
        batches = list(data_loader)
        assert len(batches) == 3
        assert ["g" * 7, "f" * 6, "e" * 5] in batches
        assert ["dddd", "ccc", "bb"] in batches
        assert ["a"] in batches

        truncating_ascending_sampler = PooledBatchSampler(
            self.data, batch_size=2, drop_last=True, key=len
        )
        data_loader = DataLoader(self.data, batch_sampler=truncating_ascending_sampler)
        batches = list(data_loader)
        assert len(batches) == 3
        assert ["a", "bb"] in batches
        assert ["ccc", "dddd"] in batches
        assert ["e" * 5, "f" * 6] in batches

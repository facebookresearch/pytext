#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import json
import random
import unittest

import numpy as np
import torch
from pytext.utils import set_random_seeds
from pytext.utils.data import (
    align_slot_labels,
    merge_token_labels_to_slot,
    strip_bio_prefix,
    unkify,
)
from pytext.utils.distributed import get_shard_range
from pytext.utils.file_io import PathManager
from pytext.utils.meter import TimeMeter
from pytext.utils.test import import_tests_module


tests_module = import_tests_module()
RAW_TEST_PATH = tests_module.test_file("test_music_samples.json")


def get_test_sample():
    with PathManager.open(RAW_TEST_PATH, "r") as f:
        data = json.load(f)
    return data


class UtilTest(unittest.TestCase):
    def test_merge_token_labels_to_slot(self):
        data = get_test_sample()
        for i in data:
            self.assertEqual(
                merge_token_labels_to_slot(i["token_ranges"], i["labels"]), i["output"]
            )
            self.assertEqual(
                merge_token_labels_to_slot(
                    i["token_ranges"],
                    [strip_bio_prefix(l) for l in i["labels"]],
                    use_bio_label=False,
                ),
                i["output"],
            )

    def test_align_slot_labels(self):
        self.assertEqual(
            align_slot_labels(
                [[0, 4], [5, 8], [9, 14], [15, 19], [20, 25]],
                "20:25:music/type,5:14:music/artistName",
                True,
            ),
            "NoLabel B-music/artistName I-music/artistName NoLabel B-music/type",
        )

    def test_align_slot_labels_with_none_label(self):
        self.assertEqual(
            align_slot_labels([[0, 4], [5, 8]], None, True), "NoLabel NoLabel"
        )

    def test_unkify(self):
        map_token_unkified = {"": "<unk>", "72": "<unk>-NUM"}

        for token, expected_unkified in map_token_unkified.items():
            self.assertEqual(unkify(token), expected_unkified)

    def test_get_shard_range(self):
        # first 5 ranks should take 3 examples
        # last 3 ranks should take 2 examples, but to make sure all shard have
        # same size, we pad with the previous example.
        dataset_size, world_size = 21, 8
        expected = [
            (0, (0, 2)),
            (1, (3, 5)),
            (2, (6, 8)),
            (3, (9, 11)),
            (4, (12, 14)),
            (5, (14, 16)),
            (6, (16, 18)),
            (7, (18, 20)),
        ]
        for rank, expected_range in expected:
            shard_range = get_shard_range(dataset_size, rank, world_size)
            self.assertEqual(shard_range, expected_range)

        dataset_size, world_size = 16, 4
        expected = [(0, (0, 3)), (1, (4, 7)), (2, (8, 11)), (3, (12, 15))]
        for rank, expected_range in expected:
            shard_range = get_shard_range(dataset_size, rank, world_size)
            self.assertEqual(shard_range, expected_range)

    def test_time_meter(self):
        tps = TimeMeter()
        for i in range(10):
            tps.update(i)
        self.assertEqual(tps.n, 45)
        self.assertTrue(tps.avg > 1)

    def test_set_random_seeds(self):
        set_random_seeds(456, False)

        self.assertEqual(random.randint(23, 57), 51)
        self.assertEqual(np.random.randint(93, 177), 120)
        self.assertTrue(
            bool(
                torch.eq(
                    torch.randint(23, 57, (1,)), torch.tensor([24], dtype=torch.long)
                ).tolist()[0]
            )
        )

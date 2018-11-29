#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import json
import os
import unittest

from pytext.utils import test_utils


RAW_TEST_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "test_samples.json"
)


def get_test_sample():
    with open(RAW_TEST_PATH, "r") as f:
        data = json.load(f)
    return data


class TestUtilTest(unittest.TestCase):
    def test_merge_token_labels_to_slot(self):
        data = get_test_sample()
        for i in data:
            self.assertEqual(
                test_utils.merge_token_labels_to_slot(i["token_ranges"], i["labels"]),
                i["output"],
            )
            self.assertEqual(
                test_utils.merge_token_labels_to_slot(
                    i["token_ranges"],
                    [test_utils.strip_bio_prefix(l) for l in i["labels"]],
                    use_bio_label=False,
                ),
                i["output"],
            )

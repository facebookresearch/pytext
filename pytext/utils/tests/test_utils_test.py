#!/usr/bin/env python3

import os
import unittest
import json

from pytext.utils import data_utils
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
                set(test_utils.merge_adjacent_token_labels_to_slots(
                    i["token_ranges"],
                    [test_utils.strip_bio_prefix(l) for l in i["labels"]],
                )),
                set(data_utils.parse_slot_string(i["output"])),
            )

    def test_merge_token_bio_labels(self):
        data = get_test_sample()
        for i in data:
            self.assertEqual(
                set(test_utils.merge_token_bio_labels_to_slots(
                    i["token_ranges"],
                    i["labels"],
                )),
                set(data_utils.parse_slot_string(i["output"])),
            )

    def test_format_label(self):
        data = get_test_sample()
        for i in data:
            self.assertEqual(
                test_utils.format_token_labels(
                    test_utils.merge_token_labels_to_slots(
                        i["token_ranges"],
                        i["labels"],
                    )
                ),
                i["output"],
            )

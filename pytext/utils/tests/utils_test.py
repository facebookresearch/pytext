#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import json
import unittest

from pytext.utils import test_utils
from pytext.utils.data_utils import align_slot_labels, unkify
from pytext.utils.test_utils import import_tests_module


tests_module = import_tests_module()
RAW_TEST_PATH = tests_module.test_file("test_music_samples.json")


def get_test_sample():
    with open(RAW_TEST_PATH, "r") as f:
        data = json.load(f)
    return data


class UtilTest(unittest.TestCase):
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
        map_token_unkified = {
            "": "<unk>",
            "Text": "<unk>-INITC",
            "TExt": "<unk>-CAPS",
            "!Text": "<unk>-CAPS",
            "text": "<unk>-LC",
            "text0": "<unk>-LC-NUM",
            "text-0": "<unk>-LC-NUM-DASH",
            "texts": "<unk>-LC-s",
            "texted": "<unk>-LC-ed",
            "texting": "<unk>-LC-ing",
            "textion": "<unk>-LC-ion",
            "texter": "<unk>-LC-er",
            "texest": "<unk>-LC-est",
            "textly": "<unk>-LC-ly",
            "textity": "<unk>-LC-ity",
            "texty": "<unk>-LC-y",
            "textal": "<unk>-LC-al",
        }

        for token, expected_unkified in map_token_unkified.items():
            self.assertEqual(unkify(token), expected_unkified)

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import glob
import json
import unittest

from pytext.builtin_task import register_builtin_tasks
from pytext.config.serialize import parse_config


register_builtin_tasks()


# These JSON files are not parseable configs
EXCLUDE_JSON = {
    # used by test_merge_token_labels_to_slot
    "pytext/utils/tests/test_samples.json"
}


class LoadAllConfigTest(unittest.TestCase):
    def test_load_all_configs(self):
        """
            Try an load all the json files in pytext to make sure we didn't
            break the config API.
        """
        print()
        for filename in glob.iglob("pytext/**/*.json", recursive=True):
            if filename in EXCLUDE_JSON:
                continue
            print("--- loading:", filename)
            with open(filename) as file:
                config_json = json.load(file)
                config = parse_config(config_json)
                self.assertIsNotNone(config)

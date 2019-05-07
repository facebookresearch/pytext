#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import glob
import json
import os
import unittest

from pytext.builtin_task import register_builtin_tasks
from pytext.config import LATEST_VERSION, PyTextConfig, config_from_json
from pytext.config.config_adapter import (
    ADAPTERS,
    upgrade_one_version,
    upgrade_to_latest,
)


register_builtin_tasks()


class ConfigAdapterTest(unittest.TestCase):
    def test_has_all_adapters(self):
        self.assertEqual(
            LATEST_VERSION,
            max(ADAPTERS.keys()) + 1,
            f"Missing adapter for LATEST_VERSION",
        )
        for i, v in enumerate(sorted(ADAPTERS.keys())):
            self.assertEqual(i, v, f"Missing adapter for version {i}")

    def test_upgrade_one_version(self):
        for p in glob.iglob(
            os.path.join(os.path.dirname(__file__), "json_config/*.json")
        ):
            print("Trying to upgrade file:" + p)
            with open(p) as f:
                test_data = json.load(f)
                for test_case in test_data:
                    old_config = test_case["original"]
                    new_config = upgrade_one_version(old_config)
                    self.assertEqual(new_config, test_case["adapted"])

    # ensure every historical config can be upgrade to latest
    def test_upgrade_to_latest(self):
        for p in glob.iglob(
            os.path.join(os.path.dirname(__file__), "json_config/*.json")
        ):
            print("Trying to upgrade file:" + p)
            with open(p) as f:
                test_data = json.load(f)
                for test_case in test_data:
                    json_config = upgrade_to_latest(test_case["original"])
                    config_from_json(PyTextConfig, json_config)

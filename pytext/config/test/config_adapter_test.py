#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import glob
import json
import os
import unittest

from pytext.builtin_task import register_builtin_tasks
from pytext.config import LATEST_VERSION, pytext_config_from_json
from pytext.config.config_adapter import (
    ADAPTERS,
    DOWNGRADE_ADAPTERS,
    upgrade_one_version,
    downgrade_one_version,
)
from pytext.utils.file_io import PathManager


register_builtin_tasks()

# Downgrade is introduced since v23
FIRST_DOWNGRADE_ADAPTER_VERSION = 23


class ConfigAdapterTest(unittest.TestCase):
    def test_all_upgrade_adapters_exist(self):
        self.assertEqual(
            LATEST_VERSION,
            max(ADAPTERS.keys()) + 1,
            "Missing adapter for LATEST_VERSION",
        )
        for i, v in enumerate(sorted(ADAPTERS.keys())):
            self.assertEqual(i, v, f"Missing upgrade adapter for version {i}")

    def test_all_downgrade_adapters_exist(self):
        """
        We need to test downgrade adapter for version >= FIRST_DOWNGRADE_ADAPTER_VERSION
        """
        for i, v in enumerate(sorted(DOWNGRADE_ADAPTERS.keys())):
            self.assertEqual(
                i + FIRST_DOWNGRADE_ADAPTER_VERSION,
                v,
                f"Missing downgrade adapter for version {v}",
            )

    # Ensure test coverage after version FIRST_DOWNGRADE_ADAPTER_VERSION
    # If the upgrade/downgrade does nothing, create a file anyways and put an
    # empty array in it
    def test_all_test_files_after_first_downgrade_adapter_version_exist(self):
        for v in range(FIRST_DOWNGRADE_ADAPTER_VERSION, LATEST_VERSION + 1):
            self.assertTrue(
                os.path.isfile(
                    os.path.join(
                        os.path.dirname(__file__), f"json_config/v{v}_test_upgrade.json"
                    )
                ),
                f"Missing upgrade test for version {v}",
            )
            self.assertTrue(
                os.path.isfile(
                    os.path.join(
                        os.path.dirname(__file__),
                        f"json_config/v{v}_test_downgrade.json",
                    )
                ),
                f"Missing downgrade test for version {v}",
            )

    def test_upgrade_one_version(self):
        # Always show the full diff, easier to debug when getting a failed log
        self.maxDiff = None
        for p in glob.iglob(
            os.path.join(os.path.dirname(__file__), "json_config/*_test_upgrade.json")
        ):
            print("Trying to upgrade file:" + p)
            with PathManager.open(p) as f:
                test_data = json.load(f)
                for test_case in test_data:
                    old_config = test_case["original"]
                    new_config = upgrade_one_version(old_config)
                    self.assertEqual(new_config, test_case["adapted"])

    def test_downgrade_one_version(self):
        # Always show the full diff, easier to debug when getting a failed log
        self.maxDiff = None
        for p in glob.iglob(
            os.path.join(os.path.dirname(__file__), "json_config/*_test_downgrade.json")
        ):
            print("Trying to downgrade file:" + p)
            with PathManager.open(p) as f:
                test_data = json.load(f)
                for test_case in test_data:
                    new_config = test_case["adapted"]
                    old_config = downgrade_one_version(new_config)
                    self.assertEqual(old_config, test_case["original"])

    # ensure every historical config can be upgrade to latest
    def test_upgrade_to_latest(self):
        for p in glob.iglob(
            os.path.join(os.path.dirname(__file__), "json_config/*.json")
        ):
            print("Trying to upgrade file:" + p + " to latest")
            with PathManager.open(p) as f:
                test_data = json.load(f)
                for test_case in test_data:
                    # make sure the config can be instantiated, don't need return value
                    pytext_config_from_json(test_case["original"])

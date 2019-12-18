#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import glob
import json
import os
import unittest

from pytext.builtin_task import register_builtin_tasks
from pytext.config.serialize import parse_config
from pytext.utils.file_io import PathManager
from pytext.utils.path import PYTEXT_HOME, get_absolute_path


register_builtin_tasks()


# These JSON files are not parseable configs
EXCLUDE_JSON = {
    # used by test_merge_token_labels_to_slot
    "utils/tests/test_samples.json"
}
# TODO: @stevenliu T52746850 include all config files from demo, include
# as many as possible from fb
EXCLUDE_DIRS = {"config/test/json_config", "tests/data", "fb", "demo"}


class LoadAllConfigTest(unittest.TestCase):
    def setUp(self):
        os.chdir(PYTEXT_HOME)

    def DISABLED_test_load_all_configs(self):
        """
            Try an load all the json files in pytext to make sure we didn't
            break the config API.
        """
        print()
        exclude_json_path = {*[get_absolute_path(p) for p in EXCLUDE_JSON]}
        exclude_json_dir = {*[get_absolute_path(p) for p in EXCLUDE_DIRS]}
        for filename in glob.iglob("./**/*.json", recursive=True):
            filepath = get_absolute_path(filename)
            if filepath in exclude_json_path:
                continue
            if any(filepath.startswith(prefix) for prefix in exclude_json_dir):
                continue
            print("--- loading:", filepath)
            with PathManager.open(filepath) as file:
                config_json = json.load(file)
                config = parse_config(config_json)
                self.assertIsNotNone(config)

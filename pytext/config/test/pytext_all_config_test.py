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
    "utils/tests/test_samples.json",
    # "pytext/data/test/data/gpt2_encoder.json",
}
# TODO: @stevenliu T52746850 include all config files from demo, include
# as many as possible from fb
EXCLUDE_DIRS = {
    "pytext/contrib",
    "pytext/config/test/json_config",
    "pytext/demo",
    "pytext/data/test/data",
    "pytext/fb",
    "pytext/tests/data",
}


class LoadAllConfigTest(unittest.TestCase):
    def test_load_all_configs(self):
        """
        Try an load all the json files in pytext to make sure we didn't
        break the config API.
        """
        for filename in glob.iglob("pytext/**/*.json", recursive=True):
            if any(f in filename for f in EXCLUDE_JSON):
                continue
            if any(d in filename for d in EXCLUDE_DIRS):
                continue
            print("--- loading:", filename)
            with PathManager.open(filename) as file:
                config_json = json.load(file)
                config = parse_config(config_json)
                self.assertIsNotNone(config)

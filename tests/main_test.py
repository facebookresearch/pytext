#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import unittest

from click.testing import CliRunner
from pytext.main import main
from pytext.utils import test
from pytext.utils.file_io import PathManager
from pytext.utils.path import PYTEXT_HOME


tests_module = test.import_tests_module()


class TestMain(unittest.TestCase):
    def setUp(self):
        os.chdir(PYTEXT_HOME)

    def run_from_command(self, args, config_filename):
        runner = CliRunner()
        config_path = os.path.join(tests_module.TEST_CONFIG_DIR, config_filename)
        with PathManager.open(config_path, "r") as f:
            config_str = f.read()
        return runner.invoke(main, args=args, input=config_str)

    def test_docnn(self):
        # train model
        result = self.run_from_command(args=["train"], config_filename="docnn.json")
        assert not result.exception, result.exception

        # export the trained model
        result = self.run_from_command(args=["export"], config_filename="docnn.json")
        assert not result.exception, result.exception

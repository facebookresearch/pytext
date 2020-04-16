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
        self.runner = CliRunner()

    def test_docnn(self):
        # train model
        result = self.runner.invoke(
            main, args=["--config-file", "demo/configs/docnn.json", "train"]
        )
        assert not result.exception, result.exception

        # export the trained model
        result = self.runner.invoke(
            main, args=["--config-file", "demo/configs/docnn.json", "export"]
        )
        print(result.output)
        assert not result.exception, result.exception

        # predict with PyTorch model
        result = self.runner.invoke(
            main,
            args=["predict-py", "--model-file", "/tmp/model.pt"],
            input='{"text": "create an alarm for 1:30 pm"}',
        )
        assert "'prediction':" in result.output, result.exception

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import json
import os
import tempfile
import unittest

from click.testing import CliRunner
from pytext.main import main
from pytext.utils import test
from pytext.utils.config_utils import MockConfigLoader
from pytext.utils.path import PYTEXT_HOME


tests_module = test.import_tests_module()


class TestMain(unittest.TestCase):
    def setUp(self):
        self.config_base_path = os.path.join(PYTEXT_HOME, "demo/configs")
        self.runner = CliRunner()

    def find_and_patch_config(self, config_filename, output_path_prefix="pytext_demo_"):
        output_base_path = tempfile.mkdtemp(prefix=output_path_prefix)
        mock_config_loader = MockConfigLoader(
            config_base_path=self.config_base_path,
            replace_paths={"/tmp": output_base_path},
        )
        config_dict = mock_config_loader.make_config(config_filename)
        return config_dict

    def test_docnn(self):
        # prepare config
        config_dict = self.find_and_patch_config("docnn.json")
        model_path = config_dict["save_snapshot_path"]
        config_json = json.dumps(config_dict)

        # train model
        result = self.runner.invoke(main, args=["--config-json", config_json, "train"])
        assert not result.exception, result.exception

        # export the trained model
        result = self.runner.invoke(main, args=["--config-json", config_json, "export"])
        print(result.output)
        assert not result.exception, result.exception

        # predict with PyTorch model
        result = self.runner.invoke(
            main,
            args=["predict-py", "--model-file", model_path],
            input='{"text": "create an alarm for 1:30 pm"}',
        )
        assert "'prediction':" in result.output, result.exception

    def test_docnn_with_export_config(self):
        # prepare config
        config_dict = self.find_and_patch_config("docnn_wo_export.json")
        model_path = config_dict["save_snapshot_path"]
        config_json = json.dumps(config_dict)

        # train model
        result = self.runner.invoke(main, args=["--config-json", config_json, "train"])
        assert not result.exception, result.exception

        # export the trained model
        result = self.runner.invoke(
            main,
            args=[
                "--config-json",
                config_json,
                "export",
                "--export-json",
                os.path.join(self.config_base_path, "export_options.json"),
            ],
        )
        print(result.output)
        assert not result.exception, result.exception

        # predict with PyTorch model
        result = self.runner.invoke(
            main,
            args=["predict-py", "--model-file", model_path],
            input='{"text": "create an alarm for 1:30 pm"}',
        )
        assert "'prediction':" in result.output, result.exception

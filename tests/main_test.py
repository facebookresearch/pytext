#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import json
import os
import tempfile
import unittest

from click.testing import CliRunner
from pytext.main import main
from pytext.tests.utils import find_and_patch_config
from pytext.utils import test
from pytext.utils.path import PYTEXT_HOME


tests_module = test.import_tests_module()


class TestMain(unittest.TestCase):
    def setUp(self):
        self.config_base_path = os.path.join(PYTEXT_HOME, "demo/configs")
        self.runner = CliRunner()

    def test_docnn(self):
        # prepare config
        config_dict = find_and_patch_config(
            "docnn.json", config_base_path=self.config_base_path
        )
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
        config_dict = find_and_patch_config(
            "docnn_wo_export.json", config_base_path=self.config_base_path
        )
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

    def test_docnn_with_torchscript_export(self):
        # prepare config
        config_dict = find_and_patch_config(
            "docnn_wo_export.json", config_base_path=self.config_base_path
        )
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
                "torchscript-export",
                "--export-json",
                os.path.join(self.config_base_path, "export_options.json"),
            ],
        )
        expected = [
            "accelerate: []",
            "batch_padding_control: None",
            "export_caffe2_path: /tmp/model.caffe2.predictor",
            "export_lite_path: None",
            "export_onnx_path: /tmp/model.onnx",
            "export_torchscript_path: /tmp/new_docnn.pt1",
            "seq_padding_control: None",
            "target: unknown",
            "torchscript_quantize: False",
        ]
        assert not result.exception, result.exception
        for item in expected:
            assert item in result.output

        # export the trained model with output path
        result = self.runner.invoke(
            main,
            args=[
                "--config-json",
                config_json,
                "torchscript-export",
                "--export-json",
                os.path.join(self.config_base_path, "export_options.json"),
                "--output-path",
                "test-path",
            ],
        )
        expected = [
            "accelerate: []",
            "batch_padding_control: None",
            "export_caffe2_path: None",
            "export_lite_path: None",
            "export_onnx_path: /tmp/model.onnx",
            "export_torchscript_path: /tmp/new_docnn.pt1",
            "seq_padding_control: None",
            "target: unknown",
            "torchscript_quantize: None",
        ]
        assert not result.exception, result.exception
        for item in expected:
            assert item in result.output

        result = self.runner.invoke(
            main,
            args=["predict-py", "--model-file", model_path],
            input='{"text": "create an alarm for 1:30 pm"}',
        )
        assert "'prediction':" in result.output, result.exception

    def test_docnn_with_torchscript_export_target(self):
        # prepare config
        config_dict = find_and_patch_config(
            "docnn_wo_export.json", config_base_path=self.config_base_path
        )
        config_json = json.dumps(config_dict)

        # train model
        result = self.runner.invoke(main, args=["--config-json", config_json, "train"])
        assert not result.exception, result.exception

        tgts = ["cpu"]
        for tgt in tgts:
            # export the trained model
            result = self.runner.invoke(
                main,
                args=[
                    "--config-json",
                    config_json,
                    "torchscript-export",
                    "--export-json",
                    os.path.join(self.config_base_path, "export_list.json"),
                    "--target",
                    tgt,
                ],
            )
            # assert the target option was accepted as legitimate.
            assert not result.exception, result.exception

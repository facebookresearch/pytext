#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import json
import os
from importlib import import_module

from pytext.config import PyTextConfig, config_to_json
from pytext.config.config_adapter import upgrade_to_latest
from pytext.config.serialize import pytext_config_from_json
from pytext.utils.file_io import PathManager


class MockConfigLoader:
    def __init__(
        self,
        config_base_path: str,
        reset_paths: {str: str} = None,
        replace_paths: {str: str} = None,
    ):
        self.config_base_path = config_base_path
        self.reset_paths = reset_paths or {}
        self.replace_paths = replace_paths or {}

    def make_config(self, config_filename, disable_tensorboard=True):
        with PathManager.open(
            os.path.join(self.config_base_path, config_filename)
        ) as config_file:
            # load config json file into dict
            config_dict = json.load(config_file)
            if "config" in config_dict:
                config_dict = config_dict["config"]
            return self.make_config_from_dict(config_dict, disable_tensorboard)

    def make_config_from_dict(self, config, disable_tensorboard):
        # config is the path module name of the actual PyText config
        if isinstance(config, str):
            config = config_to_json(PyTextConfig, import_module(config).config)
        config = upgrade_to_latest(config)
        # Disable TensorBoard for integration tests
        if disable_tensorboard:
            config["use_tensorboard"] = False
        return self.disable_cuda(self.fix_paths(config))

    def disable_cuda(self, config):
        config["use_cuda_if_available"] = False
        if "distributed_world_size" in config:
            del config["distributed_world_size"]
        return config

    def fix_paths(self, config):
        if isinstance(config, str):
            for src, dest in self.replace_paths.items():
                config = config.replace(src, dest)
            return config
        elif isinstance(config, dict):
            return {key: self.fix_paths(value) for key, value in config.items()}
        elif isinstance(config, list):
            return [self.fix_paths(element) for element in config]
        else:
            return config

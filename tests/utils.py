#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import tempfile

from pytext.utils import test
from pytext.utils.config_utils import MockConfigLoader

tests_module = test.import_tests_module()


def find_and_patch_config(
    config_filename, config_base_path, output_path_prefix="pytext_demo_"
):
    output_base_path = tempfile.mkdtemp(prefix=output_path_prefix)
    mock_config_loader = MockConfigLoader(
        config_base_path=config_base_path,
        replace_paths={"/tmp": output_base_path},
    )
    config_dict = mock_config_loader.make_config(config_filename)
    return config_dict

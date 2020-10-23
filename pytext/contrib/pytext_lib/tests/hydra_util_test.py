#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from pytext.contrib.pytext_lib.utils.hydra_util import to_container


class TestHydraUtil(unittest.TestCase):
    def test_to_container_list(self):
        conf = OmegaConf.create([1, 2, 3])
        self.assertIsInstance(conf, ListConfig)
        real_obj = to_container(conf)
        self.assertIsInstance(real_obj, list)

    def test_to_container_dict(self):
        conf = OmegaConf.create({"a": 1, "b": "c"})
        self.assertIsInstance(conf, DictConfig)
        real_obj = to_container(conf)
        self.assertIsInstance(real_obj, dict)

    def test_to_container_any(self):
        conf = [1, 2, 3]
        real_obj = to_container(conf)
        self.assertIsInstance(real_obj, list)

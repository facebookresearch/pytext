#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import unittest
from dataclasses import dataclass

from omegaconf import OmegaConf
from pytext.contrib.pytext_lib.utils.data_util import to_omega_conf


@dataclass
class SampleDataclass:
    field1: int
    field2: str


DICT_SAMPLE = {"field1": 42, "field2": "abc"}


class TestDataUtil(unittest.TestCase):
    def test_to_omega_conf_omega(self):
        conf = OmegaConf.create(DICT_SAMPLE)
        self.assertEqual(to_omega_conf(conf), conf)

    def test_to_omega_conf_dataclass(self):
        conf = to_omega_conf(SampleDataclass(**DICT_SAMPLE))
        self.assertEqual(conf.field1, DICT_SAMPLE["field1"])
        self.assertEqual(conf.field2, DICT_SAMPLE["field2"])

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest
from typing import Union

from pytext.config import serialize


SAMPLE_INT_JSON = {"int": 6}
SAMPLE_UNION_CLS = Union[str, int]


class SerializeTest(unittest.TestCase):
    def test_value_from_json(self):
        print()
        value = serialize._value_from_json(SAMPLE_UNION_CLS, SAMPLE_INT_JSON)
        self.assertEqual(6, value)

    def test_value_to_json(self):
        print()
        json = serialize._value_to_json(Union[str, int], 6)
        self.assertEqual(SAMPLE_INT_JSON, json)

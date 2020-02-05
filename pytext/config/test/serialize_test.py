#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest
from typing import Any, Dict, Union

from pytext.config import serialize


SAMPLE_INT_JSON = {"int": 6}
SAMPLE_UNION_CLS = Union[str, int]

SAMPLE_DICT_WITH_ANY_CLS = Dict[str, Any]
DICT_WITH_ANY: SAMPLE_DICT_WITH_ANY_CLS = {"lr": 0.1, "type": "FedAvg"}
SAMPLE_DICT_WITH_ANY_JSON = {"lr": {"float": 0.1}, "type": {"str": "FedAvg"}}

SAMPLE_ANY_CLS = Any
MULTI_TYPE_LIST: Any = [1, "test", 0.01]
SAMPLE_ANY: Any = {"list": MULTI_TYPE_LIST}


class SerializeTest(unittest.TestCase):
    def test_value_from_json(self):
        print()
        value = serialize._value_from_json(SAMPLE_UNION_CLS, SAMPLE_INT_JSON)
        self.assertEqual(6, value)

    def test_value_to_json(self):
        print()
        json = serialize._value_to_json(Union[str, int], 6)
        self.assertEqual(SAMPLE_INT_JSON, json)

    def test_value_to_json_for_class_type_any(self):
        json = serialize._value_to_json(Dict[str, Any], DICT_WITH_ANY)
        self.assertEqual(json, SAMPLE_DICT_WITH_ANY_JSON)

        json = serialize._value_to_json(Any, [1, "test", 0.01])
        self.assertEqual(json, SAMPLE_ANY)

    def test_value_from_json_for_class_type_any(self):
        value = serialize._value_from_json(
            SAMPLE_DICT_WITH_ANY_CLS, SAMPLE_DICT_WITH_ANY_JSON
        )
        self.assertEqual(DICT_WITH_ANY, value)

        value = serialize._value_from_json(SAMPLE_ANY_CLS, SAMPLE_ANY)
        self.assertEqual([1, "test", 0.01], value)

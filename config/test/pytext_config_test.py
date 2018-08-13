#!/usr/bin/env python3

import json
import unittest
from typing import Union

from pytext.config.pytext_config import ConfigBase
from pytext.config.serialize import (
    ConfigParseError,
    MissingValueError,
    UnionTypeError,
    config_from_json,
)


class Model1(ConfigBase):
    foo: int = 1


class Model2(ConfigBase):
    bar: str


class Model2Sub1(Model2, ConfigBase):
    m2s1: int = 5


class Model2Sub1Sub1(Model2Sub1, ConfigBase):
    m2s1s1: str
    bar: int = 3  # noqa


class Model3(ConfigBase):
    foobar: float


class Task1(ConfigBase):
    model: Union[Model1, Model2]


class Task2(ConfigBase):
    model: Model1


class PyTextConfig(ConfigBase):
    task: Union[Task1, Task2]
    output: str


class ConfigBaseTest(unittest.TestCase):
    def test_inherit(self):
        self.assertEqual(len(Model2Sub1Sub1.__annotations__), 3)
        self.assertEqual(
            Model2Sub1Sub1.__annotations__["bar"],
            int,
            "fields in subclass should overwrite same ones in baseclass",
        )
        self.assertEqual(
            Model2Sub1._field_defaults["m2s1"],
            5,
            "default value of fields in baseclass should be inherited",
        )
        self.assertEqual(
            Model2Sub1Sub1._field_defaults["bar"],
            3,
            "default value of fields in subclass should overwrite same ones in baseclass",
        )
        obj = Model2Sub1Sub1(m2s1s1='test')
        self.assertEqual(obj.m2s1, 5)


class PytextConfigTest(unittest.TestCase):
    def test_use_default_value(self):
        config_json = json.loads(
            """
            {}
        """
        )
        config = config_from_json(Model1, config_json)
        self.assertEqual(config.foo, 1)

    def test_nested_config_with_union(self):
        config_json = json.loads(
            """
            {
                "task": {
                    "task1": {
                        "model": {
                            "model2": {
                                "bar": "test"
                            }
                        }
                    }
                },
                "output": "foo/bar.pt"
            }
        """
        )
        config = config_from_json(PyTextConfig, config_json)
        self.assertTrue(isinstance(config.task, Task1))
        self.assertTrue(isinstance(config.task.model, Model2))
        self.assertEqual(config.task.model.bar, "test")

    def test_missing_value(self):
        config_json = json.loads(
            """
            {}
        """
        )
        self.assertRaises(MissingValueError, config_from_json, Model2, config_json)

    def test_incorrect_union(self):
        config_json = json.loads(
            """
            {
                "task": {
                    "task1": {
                        "model": {
                            "model3": {

                            }
                        }
                    }
                },
                "output": "test"
            }
            """
        )
        self.assertRaises(UnionTypeError, config_from_json, PyTextConfig, config_json)

    def test_incorrect_input_type(self):
        config_json = json.loads(
            """
            {"bar": 123}
        """
        )
        self.assertRaises(ConfigParseError, config_from_json, Model2, config_json)

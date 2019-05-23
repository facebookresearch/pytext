#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import json
import unittest
from typing import List, Tuple, Union

from pytext.config.component import Component, ComponentType
from pytext.config.pytext_config import ConfigBase
from pytext.config.serialize import (
    ConfigParseError,
    MissingValueError,
    UnionTypeError,
    config_from_json,
    config_to_json,
)


class Model1(ConfigBase):
    foo: int = 1


class Model2(ConfigBase):
    bar: str


class Model2Sub1(Model2):
    m2s1: int = 5


class Model2Sub1Sub1(Model2Sub1):
    m2s1s1: str
    bar: int = 3


class Model3(ConfigBase):
    foobar: float


class ExpansibleModel(Component):
    __COMPONENT_TYPE__ = ComponentType.MODEL
    __EXPANSIBLE__ = True

    class Config(ConfigBase):
        foobar: float


class AExpansibleModel(ExpansibleModel):
    class Config(ExpansibleModel.Config):
        foo: int


class BExpansibleModel(ExpansibleModel):
    class Config(ExpansibleModel.Config):
        bar: str


class Task1(ConfigBase):
    model: Union[Model1, Model2]


class Task2(ConfigBase):
    model: Model1


class Task3(ConfigBase):
    model: ExpansibleModel.Config


class PyTextConfig(ConfigBase):
    task: Union[Task1, Task2, Task3]
    output: str


class RegisteredModel(Component):
    __COMPONENT_TYPE__ = ComponentType.MODEL
    Config = Model2Sub1Sub1


class JointModel(Component):
    __COMPONENT_TYPE__ = ComponentType.MODEL

    class Config(Model1):
        models: List[RegisteredModel.Config]


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
        obj = Model2Sub1Sub1(m2s1s1="test")
        self.assertEqual(obj.m2s1, 5)
        self.assertEqual(len(obj.items()), 3)

    def test_subclassing_valid_ordering(self):
        class SubclassDefaultOrdering(Model1):
            foo: int
            bar: int

        class SubclassDefaultOrdering2(SubclassDefaultOrdering, Model1):
            pass


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
                    "Task1": {
                        "model": {
                            "Model2": {
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

    def test_default_union(self):
        config_json = json.loads(
            """
            {
                "task": {
                    "model": {
                        "Model2": {
                            "bar": "test"
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

    def test_default_value_expansible_config(self):
        config_json = json.loads(
            """
            {
                "task": {
                    "Task3": {
                        "model": {
                            "foobar": 1.0
                        }
                    }
                },
                "output": "foo/bar.pt"
            }
        """
        )
        config = config_from_json(PyTextConfig, config_json)
        self.assertTrue(isinstance(config.task, Task3))
        self.assertTrue(isinstance(config.task.model, ExpansibleModel.Config))
        self.assertFalse(isinstance(config.task.model, AExpansibleModel.Config))
        self.assertFalse(isinstance(config.task.model, BExpansibleModel.Config))
        self.assertEqual(config.task.model.foobar, 1.0)

    def test_component(self):
        config_json = json.loads(
            """{
            "bar": 13,
            "m2s1s1": "sub_foo"
        }"""
        )
        config = config_from_json(RegisteredModel.Config, config_json)
        self.assertEqual(config.bar, 13)
        self.assertEqual(config.m2s1s1, "sub_foo")
        self.assertEqual(config.m2s1, 5)

    def test_component_subconfig_serialize(self):
        config_json = json.loads(
            """{
            "foo": 5,
            "models": [{
                "bar": 12,
                "m2s1s1": "thing"
            }, {
                "m2s1s1": "thing2"
            }]
        }"""
        )
        config = config_from_json(JointModel.Config, config_json)
        serialized = config_to_json(JointModel.Config, config)
        again = config_from_json(JointModel.Config, serialized)
        self.assertEqual(again.foo, 5)
        self.assertEqual(again.models[0].m2s1s1, "thing")
        self.assertEqual(again.models[1].bar, 3)

    def test_component_subconfig_deserialize(self):
        config_json = json.loads(
            """{
            "foo": 5,
            "models": [{
                "bar": 12,
                "m2s1s1": "thing"
            }, {
                "m2s1s1": "thing2"
            }]
        }"""
        )
        config = config_from_json(JointModel.Config, config_json)
        self.assertEqual(config.foo, 5)
        self.assertEqual(len(config.models), 2)
        self.assertEqual(config.models[1].m2s1s1, "thing2")

    def test_missing_value(self):
        config_json = json.loads(
            """
            {}
        """
        )
        self.assertRaises(MissingValueError, config_from_json, Model2, config_json)

    def test_unknown_fields(self):
        config_json = json.loads(
            """{
            "FAKE_FIELD": 7
        }"""
        )
        self.assertRaises(ConfigParseError, config_from_json, Model2, config_json)

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
            {"foo": "abc"}
        """
        )
        self.assertRaises(ConfigParseError, config_from_json, Model1, config_json)

    def test_implicit_type_cast(self):
        config_json = json.loads(
            """
            {"foo": "123"}
        """
        )
        config = config_from_json(Model1, config_json)
        self.assertEqual(config.foo, 123)


class TupleTestConfig(ConfigBase):
    foo: Tuple[Tuple[int, int], ...] = ((32, 2),) * 2
    bar: Tuple[int, str] = (1, "test")


class TupleConfigTest(unittest.TestCase):
    def test_invalid_tuple(self):
        config_json = json.loads(
            """
                {"bar": ["test", "test"]}
            """
        )
        self.assertRaises(
            ConfigParseError, config_from_json, TupleTestConfig, config_json
        )

    def test_invalid_multiple_values_with_no_ellipsis(self):
        config_json = json.loads(
            """
                {"bar": [1, "test", 2]}
            """
        )
        self.assertRaises(
            ConfigParseError, config_from_json, TupleTestConfig, config_json
        )

    def test_nested_tuple(self):
        config_json = json.loads(
            """
                {}
            """
        )
        config = config_from_json(TupleTestConfig, config_json)
        self.assertEqual(config.foo[0][0], 32)
        self.assertEqual(config.foo[1][1], 2)

    def test_multiple_values_with_ellipsis(self):
        config_json = json.loads(
            """
                {"foo":[[1,2], [3,4], [5,6]]}
            """
        )
        config = config_from_json(TupleTestConfig, config_json)
        self.assertEqual(config.foo[0][0], 1)
        self.assertEqual(config.foo[2][1], 6)

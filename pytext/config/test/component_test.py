#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import json
import unittest
from typing import Union

from pytext.config.component import Component, ComponentType, ConfigBase
from pytext.config.serialize import ConfigParseError, config_from_json, config_to_json


class Model(Component):
    __COMPONENT_TYPE__ = ComponentType.MODEL
    __EXPANSIBLE__ = True


class ModelFoo(Model):
    class Config(ConfigBase):
        foo: int = 2


class ModelFoo1(ModelFoo):
    pass


class ModelBar(Model):
    __EXPANSIBLE__ = True

    class Config(ConfigBase):
        bar: str = "bar"


class ModelBar1(ModelBar):
    class Config(ConfigBase):
        bar: str = "bar1"


class ModelBar2(ModelBar):
    class Config(ConfigBase):
        bar: str = "bar2"


class DataHandler(Component):
    __COMPONENT_TYPE__ = ComponentType.DATA_HANDLER
    __EXPANSIBLE__ = True


class SubDataHandler(DataHandler):
    class Config(ConfigBase):
        foo: int = 3


class TestConfig(ConfigBase):
    model: Model.Config
    model_bar_x: ModelBar.Config
    model_union: Union[ModelFoo.Config, ModelBar.Config]
    datahandler: DataHandler.Config


class IncorrectTestConfig(ConfigBase):
    model: ModelBar.Config


class NonExpansibleTestConfig(ConfigBase):
    model: ModelFoo.Config


class ComponentTest(unittest.TestCase):
    def test_auto_include_sub_component_in_config(self):
        config_json = json.loads(
            """
            {
                "model": {"ModelFoo": {}},
                "model_bar_x": {"ModelBar1": {}},
                "model_union": {"ModelBar": {}},
                "datahandler": {"SubDataHandler": {}}
            }
        """
        )
        config = config_from_json(TestConfig, config_json)
        self.assertEqual(config.model.foo, 2)
        self.assertEqual(config.model_bar_x.bar, "bar1")
        self.assertEqual(config.model_union.bar, "bar")
        self.assertEqual(config.datahandler.foo, 3)

    def test_use_non_sub_component_in_config(self):
        config_json = json.loads(
            """
            {
                "model": {"ModelFoo": {}}
            }
        """
        )
        self.assertRaises(
            ConfigParseError, config_from_json, IncorrectTestConfig, config_json
        )

    def test_use_non_expansible_component_in_config(self):
        config_json = json.loads(
            """
            {
                "model": {"ModelFoo1": {}}
            }
        """
        )
        self.assertRaises(
            ConfigParseError, config_from_json, NonExpansibleTestConfig, config_json
        )

    def test_serialize_expansible_component(self):
        config = TestConfig(
            model=ModelFoo.Config(),
            model_bar_x=ModelBar1.Config(),
            model_union=ModelBar.Config(),
            datahandler=SubDataHandler.Config(),
        )
        json = config_to_json(TestConfig, config)
        print(json)
        self.assertEqual(json["model"]["ModelFoo"]["foo"], 2)
        self.assertEqual(json["model_bar_x"]["ModelBar1"]["bar"], "bar1")
        self.assertEqual(json["model_union"]["ModelBar"]["bar"], "bar")
        self.assertEqual(json["datahandler"]["SubDataHandler"]["foo"], 3)
        config = config_from_json(TestConfig, json)
        self.assertEqual(config.model.foo, 2)
        self.assertEqual(config.model_bar_x.bar, "bar1")
        self.assertEqual(config.model_union.bar, "bar")
        self.assertEqual(config.datahandler.foo, 3)

    def test_serialize_union_with_expansible_component(self):
        config = TestConfig(
            model=ModelFoo.Config(),
            model_bar_x=ModelBar.Config(),
            model_union=ModelBar1.Config(),
            datahandler=SubDataHandler.Config(),
        )
        json = config_to_json(TestConfig, config)
        print(json)
        self.assertEqual(json["model"]["ModelFoo"]["foo"], 2)
        self.assertEqual(json["model_bar_x"]["ModelBar"]["bar"], "bar")
        self.assertEqual(json["model_union"]["ModelBar1"]["bar"], "bar1")
        self.assertEqual(json["datahandler"]["SubDataHandler"]["foo"], 3)
        config = config_from_json(TestConfig, json)
        self.assertEqual(config.model.foo, 2)
        self.assertEqual(config.model_bar_x.bar, "bar")
        self.assertEqual(config.model_union.bar, "bar1")
        self.assertEqual(config.datahandler.foo, 3)

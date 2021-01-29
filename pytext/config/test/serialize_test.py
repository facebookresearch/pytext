#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest
from typing import Any, Dict, Union, List

from pytext.config import ConfigBase, pytext_config_from_json, serialize


SAMPLE_INT_JSON = {"int": 6}
SAMPLE_UNION_CLS = Union[str, int]

SAMPLE_DICT_WITH_ANY_CLS = Dict[str, Any]
DICT_WITH_ANY: SAMPLE_DICT_WITH_ANY_CLS = {"lr": 0.1, "type": "FedAvg"}

SAMPLE_ANY_CLS = Any
MULTI_TYPE_LIST: Any = [1, "test", 0.01]
SAMPLE_ANY: Any = MULTI_TYPE_LIST


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
        self.assertEqual(json, DICT_WITH_ANY)

        json = serialize._value_to_json(Any, [1, "test", 0.01])
        self.assertEqual(json, SAMPLE_ANY)

    def test_value_from_json_for_class_type_any(self):
        value = serialize._value_from_json(SAMPLE_DICT_WITH_ANY_CLS, DICT_WITH_ANY)
        self.assertEqual(DICT_WITH_ANY, value)

        value = serialize._value_from_json(SAMPLE_ANY_CLS, SAMPLE_ANY)
        self.assertEqual([1, "test", 0.01], value)

    def test_config_to_json_for_dict(self):
        """For a config that contains a dict inside it, verify that config
        can be correctly created from a json/dict, and config can be correctly
        serialized and de-serialized
        """

        class TestConfigContainer(ConfigBase):
            class TestConfig(ConfigBase):
                a_dict: Dict[str, Any]

            test_config: TestConfig

        a_dict = {"param2": {"nested_param1": 10, "nested_param2": 2}, "param1": "val1"}
        a_config_containing_dict = {"test_config": {"a_dict": a_dict}}
        pytext_config = serialize.config_from_json(
            TestConfigContainer, a_config_containing_dict
        )
        # verify that a_dict was read correctly
        self.assertEqual(pytext_config.test_config.a_dict, a_dict)
        # serialize config to json, and deserialize back to config
        # verify that nothing changed
        jsonified_config = serialize.config_to_json(TestConfigContainer, pytext_config)
        pytext_config_deserialized = serialize.config_from_json(
            TestConfigContainer, jsonified_config
        )
        self.assertEqual(pytext_config, pytext_config_deserialized)

    def test_config_to_json_for_union(self):
        """For a config that contains a dict inside it, verify that config
        can be correctly created from a json/dict, and config can be correctly
        serialized and de-serialized
        """

        class Boo(ConfigBase):
            aye: str

        class TestConfigContainer(ConfigBase):
            class TestConfig(ConfigBase):
                a_union: Union[Boo, List[Boo]]

            test_config: TestConfig

        a_str = "abc"
        a_config_containing_union = {
            "test_config": {"a_union": {"boo": {"aye": a_str}}}
        }
        pytext_config = serialize.config_from_json(
            TestConfigContainer, a_config_containing_union
        )
        # verify that a_dict was read correctly
        self.assertEqual(pytext_config.test_config.a_union.aye, a_str)
        # serialize config to json, and deserialize back to config
        # verify that nothing changed
        jsonified_config = serialize.config_to_json(TestConfigContainer, pytext_config)
        pytext_config_deserialized = serialize.config_from_json(
            TestConfigContainer, jsonified_config
        )
        self.assertEqual(pytext_config, pytext_config_deserialized)

    def test_config_to_json_for_union_with_baa(self):
        """For a config that contains a dict inside it, verify that config
        can be correctly created from a json/dict, and config can be correctly
        serialized and de-serialized
        """

        class Boo(ConfigBase):
            aye: str

        class Baa(ConfigBase):
            oye: int

        class TestConfigContainer(ConfigBase):
            class TestConfig(ConfigBase):
                a_union: Union[Boo, Baa]

            test_config: TestConfig

        an_int = 1
        a_baa = {"baa": {"oye": an_int}}
        a_config_containing_union = {"test_config": {"a_union": a_baa}}
        pytext_config = serialize.config_from_json(
            TestConfigContainer, a_config_containing_union
        )
        # verify that a_dict was read correctly
        self.assertEqual(pytext_config.test_config.a_union.oye, an_int)
        # serialize config to json, and deserialize back to config
        # verify that nothing changed
        jsonified_config = serialize.config_to_json(TestConfigContainer, pytext_config)
        pytext_config_deserialized = serialize.config_from_json(
            TestConfigContainer, jsonified_config
        )
        self.assertEqual(pytext_config, pytext_config_deserialized)

    def test_config_to_json_for_union_with_list_a(self):
        """For a config that contains a dict inside it, verify that config
        can be correctly created from a json/dict, and config can be correctly
        serialized and de-serialized
        """

        class Boo(ConfigBase):
            aye: str

        class Boolist(ConfigBase):
            boolistitems: List[Boo]

        class TestConfigContainer(ConfigBase):
            class TestConfig(ConfigBase):
                a_union: Union[Boo, Boolist]

            test_config: TestConfig

        a_str = "abc"
        # a_boo = {"boo": {"aye": a_union}}
        a_boo = {"aye": a_str}
        a_boo_list = {"boolist": {"boolistitems": [a_boo]}}
        a_config_containing_union = {"test_config": {"a_union": a_boo_list}}
        pytext_config = serialize.config_from_json(
            TestConfigContainer, a_config_containing_union
        )
        # verify that a_dict was read correctly
        self.assertEqual(pytext_config.test_config.a_union.boolistitems[0].aye, a_str)
        # serialize config to json, and deserialize back to config
        # verify that nothing changed
        jsonified_config = serialize.config_to_json(TestConfigContainer, pytext_config)
        pytext_config_deserialized = serialize.config_from_json(
            TestConfigContainer, jsonified_config
        )
        self.assertEqual(pytext_config, pytext_config_deserialized)

    def dont_test_config_to_json_for_union_with_list_b(self):
        """For a config that contains a dict inside it, verify that config
        can be correctly created from a json/dict, and config can be correctly
        serialized and de-serialized
        """

        class Boo(ConfigBase):
            aye: str

        class Boolist(ConfigBase):
            boolistitems: List[Boo]

        class TestConfigContainer(ConfigBase):
            class TestConfig(ConfigBase):
                a_union: Union[Boo, Boolist]

            test_config: TestConfig

        a_str = "abc"
        a_boo = {"boo": {"aye": a_str}}
        # a_boo = {"aye": a_str}
        a_boo_list = {"boolist": {"boolistitems": [a_boo]}}
        a_config_containing_union = {"test_config": {"a_union": a_boo_list}}
        pytext_config = serialize.config_from_json(
            TestConfigContainer, a_config_containing_union
        )
        # verify that a_dict was read correctly
        self.assertEqual(pytext_config.test_config.a_union.boolistitems[0].aye, a_str)
        # serialize config to json, and deserialize back to config
        # verify that nothing changed
        jsonified_config = serialize.config_to_json(TestConfigContainer, pytext_config)
        pytext_config_deserialized = serialize.config_from_json(
            TestConfigContainer, jsonified_config
        )
        self.assertEqual(pytext_config, pytext_config_deserialized)

    def dont_test_config_to_json_for_list_of_boo1(self):
        """For a config that contains a dict inside it, verify that config
        can be correctly created from a json/dict, and config can be correctly
        serialized and de-serialized
        """

        class Boo1(ConfigBase):
            aye: str

        class Boolist1(ConfigBase):
            boolistitems: List[Boo1]

        class TestConfigContainer(ConfigBase):
            class TestConfig(ConfigBase):
                a_list: Boolist1

            test_config: TestConfig

        a_str = "abc"
        # a_boo = {"boo1": {"aye": a_str}}
        a_boo = {"aye": a_str}
        a_boo_list = {"boolist1": {"boolistitems": [a_boo]}}
        a_config_containing_union = {"test_config": {"a_list": a_boo_list}}
        pytext_config = serialize.config_from_json(
            TestConfigContainer, a_config_containing_union
        )
        # verify that a_dict was read correctly
        self.assertEqual(pytext_config.test_config.a_list.boolistitems[0].aye, a_str)
        # serialize config to json, and deserialize back to config
        # verify that nothing changed
        jsonified_config = serialize.config_to_json(TestConfigContainer, pytext_config)
        pytext_config_deserialized = serialize.config_from_json(
            TestConfigContainer, jsonified_config
        )
        self.assertEqual(pytext_config, pytext_config_deserialized)

    def test_config_to_json_for_list_of_lboo1(self):
        """For a config that contains a dict inside it, verify that config
        can be correctly created from a json/dict, and config can be correctly
        serialized and de-serialized
        """

        class Boo1(ConfigBase):
            aye: str

        class TestConfigContainer(ConfigBase):
            class TestConfig(ConfigBase):
                a_list: List[Boo1]

            test_config: TestConfig

        a_str = "abc"
        # a_boo = {"boo1": {"aye": a_str}}
        a_boo = {"aye": a_str}
        # a_boo_list = {"boolist1": {"boolistitems": [a_boo]}}
        a_config_containing_union = {"test_config": {"a_list": [a_boo]}}
        pytext_config = serialize.config_from_json(
            TestConfigContainer, a_config_containing_union
        )
        # verify that a_dict was read correctly
        self.assertEqual(pytext_config.test_config.a_list[0].aye, a_str)
        # serialize config to json, and deserialize back to config
        # verify that nothing changed
        jsonified_config = serialize.config_to_json(TestConfigContainer, pytext_config)
        pytext_config_deserialized = serialize.config_from_json(
            TestConfigContainer, jsonified_config
        )
        self.assertEqual(pytext_config, pytext_config_deserialized)

    def test_config_to_json_for_list_of_lboo2(self):
        """For a config that contains a dict inside it, verify that config
        can be correctly created from a json/dict, and config can be correctly
        serialized and de-serialized
        """

        class Boo1(ConfigBase):
            aye: str

        class TestConfigContainer(ConfigBase):
            class TestConfig(ConfigBase):
                a_list: List[Boo1]

            test_config: TestConfig

        a_str = "abc"
        b_str = "def"
        # a_boo = {"boo1": {"aye": a_str}}
        a_boo = {"aye": a_str}
        b_boo = {"aye": b_str}
        # a_boo_list = {"boolist1": {"boolistitems": [a_boo]}}
        a_config_containing_union = {"test_config": {"a_list": [a_boo, b_boo]}}
        pytext_config = serialize.config_from_json(
            TestConfigContainer, a_config_containing_union
        )
        # verify that a_dict was read correctly
        self.assertEqual(pytext_config.test_config.a_list[0].aye, a_str)
        self.assertEqual(pytext_config.test_config.a_list[1].aye, b_str)
        # serialize config to json, and deserialize back to config
        # verify that nothing changed
        jsonified_config = serialize.config_to_json(TestConfigContainer, pytext_config)
        pytext_config_deserialized = serialize.config_from_json(
            TestConfigContainer, jsonified_config
        )
        self.assertEqual(pytext_config, pytext_config_deserialized)

    def test_config_to_json_for_nested_objectsboo2(self):
        """For a config that contains a dict inside it, verify that config
        can be correctly created from a json/dict, and config can be correctly
        serialized and de-serialized
        """

        class Boo3(ConfigBase):
            ayeaye: str

        class Boo2(ConfigBase):
            aye: Boo3

        class TestConfigContainer(ConfigBase):
            class TestConfig(ConfigBase):
                a_list: List[Boo2]

            test_config: TestConfig

        a_str = "abc"
        # a_boo = {"boo1": {"aye": a_str}}
        a_boo = {"ayeaye": a_str}
        # b_boo = {"aye": {"boo3": a_boo}}
        b_boo = {"aye": a_boo}
        # a_boo_list = {"boolist1": {"boolistitems": [a_boo]}}
        a_config_containing_union = {"test_config": {"a_list": [b_boo]}}
        pytext_config = serialize.config_from_json(
            TestConfigContainer, a_config_containing_union
        )
        # verify that a_dict was read correctly
        self.assertEqual(pytext_config.test_config.a_list[0].aye.ayeaye, a_str)
        # serialize config to json, and deserialize back to config
        # verify that nothing changed
        jsonified_config = serialize.config_to_json(TestConfigContainer, pytext_config)
        pytext_config_deserialized = serialize.config_from_json(
            TestConfigContainer, jsonified_config
        )
        self.assertEqual(pytext_config, pytext_config_deserialized)

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from enum import Enum
from typing import Any, Dict, List, Tuple, Union

from pytext.common.utils import eprint

from .component import Registry
from .config_adapter import upgrade_to_latest
from .pytext_config import PyTextConfig


class ConfigParseError(Exception):
    pass


class UnionTypeError(ConfigParseError):
    pass


class EnumTypeError(ConfigParseError):
    pass


class MissingValueError(ConfigParseError):
    pass


class IncorrectTypeError(Exception):
    pass


def _canonical_typename(cls):
    if "_name" in dir(cls):
        name = cls._name
    else:
        name = cls.__name__
    if name.endswith(".Config"):
        return name[: -len(".Config")]
    return name


def _extend_tuple_type(cls, value):
    sub_cls_list = list(cls.__args__)
    if len(sub_cls_list) != len(value):
        if len(sub_cls_list) != 2 or sub_cls_list[1] is not Ellipsis:
            raise ConfigParseError(
                f"{len(value)} values found which is more than number of types in tuple {cls}"
            )
        del sub_cls_list[1]
        sub_cls_list.extend((cls.__args__[0],) * (len(value) - len(sub_cls_list)))
    return sub_cls_list


def build_subclass_dict(subclasses):
    subclasses_dict = {}
    for subclass in subclasses:
        if type(None) != subclass:
            if getattr(subclass, "__EXPANSIBLE__", False):
                children = Registry.subconfigs(subclass)
                for child in children:
                    subclasses_dict[_canonical_typename(child).lower()] = child
            else:
                subclasses_dict[_canonical_typename(subclass).lower()] = subclass
    return subclasses_dict


def _any_from_json(json_obj):
    if _is_dict(json_obj):
        # convert to vanilla python dict
        # call _any_from_json recursively on values
        # Example: fancy_dict(a_key, fancy_dict(another_key, value))
        # will get converted into plain dict {a_key: {another_key: value}}
        return {key: _any_from_json(value) for key, value in json_obj.items()}
    return json_obj


def _union_from_json(subclasses, json_obj):
    if not _is_dict(json_obj):
        raise IncorrectTypeError(
            f"incorrect Union value {json_obj} for union {subclasses}"
        )
    subclasses_dict = build_subclass_dict(subclasses)

    type_name = list(json_obj)[0].lower()
    if len(json_obj) == 1 and type_name in subclasses_dict:
        json_obj = next(iter(json_obj.values()))
    else:
        type_name = next(iter(subclasses_dict))
        eprint(
            "WARNING - Can not find class type in json: "
            f"trying with first class {type_name} in the union."
        )
    try:
        return _value_from_json(subclasses_dict[type_name], json_obj)
    except Exception as e:
        raise UnionTypeError(
            (
                f"failed to parse union {subclasses} from"
                f"json payload {json_obj} \n"
                f"Reason: {e}"
            )
        ) from e


def _is_optional(cls):
    return _get_class_type(cls) == Union and type(None) in cls.__args__


def _enum_from_json(enum_cls, json_obj):
    for e in enum_cls:
        if e.value == json_obj:
            return e
    raise EnumTypeError(f"invalid enum value {json_obj} for {enum_cls}")


def _value_from_json(cls, value):
    cls_type = _get_class_type(cls)
    if value is None:
        return value
    # Unions must be first because Union explicitly doesn't
    # support __subclasscheck__.
    # optional with more than 2 classes is treated as Union
    elif _is_optional(cls) and len(cls.__args__) == 2:
        sub_cls = cls.__args__[0] if type(None) != cls.__args__[0] else cls.__args__[1]
        return _value_from_json(sub_cls, value)
    # nested config
    elif hasattr(cls, "_fields"):
        return config_from_json(cls, value)
    elif cls_type == Any:
        return _any_from_json(value)
    elif cls_type == Union:
        return _union_from_json(cls.__args__, value)
    elif issubclass(cls_type, Enum):
        return _enum_from_json(cls, value)
    elif issubclass(cls_type, List):
        sub_cls = cls.__args__[0]
        return [_value_from_json(sub_cls, v) for v in value]
    elif issubclass(cls_type, Tuple):
        return tuple(
            _value_from_json(c, v)
            for c, v in zip(_extend_tuple_type(cls, value), value)
        )
    elif issubclass(cls_type, Dict):
        sub_cls = cls.__args__[1]
        return {key: _value_from_json(sub_cls, v) for key, v in value.items()}
    # built in types
    return cls(value)


def _is_type_specifier(value):
    if not _is_dict(value) or len(value) != 1:
        return False
    name = next(iter(value))
    return name[0] == name[0].upper()


def _try_component_config_from_json(cls, value):
    if _is_type_specifier(value):
        options = Registry.subconfigs(cls)
        type_name = list(value)[0]
        for option in options:
            if type_name.lower() == _canonical_typename(option).lower():
                return _value_from_json(option, value[type_name])
        else:
            raise Exception(f"could not find specified component class {type_name}")
    return None


def pytext_config_from_json(json_obj, ignore_fields=(), auto_upgrade=True):
    if auto_upgrade:
        json_obj = upgrade_to_latest(json_obj)
    return config_from_json(PyTextConfig, json_obj, ignore_fields)


def config_from_json(cls, json_obj, ignore_fields=()):
    if getattr(cls, "__EXPANSIBLE__", False):
        component_config = _try_component_config_from_json(cls, json_obj)
        if component_config:
            return component_config
    parsed_dict = {}
    if not hasattr(cls, "_fields"):
        raise IncorrectTypeError(f"{cls} is not a valid config class")
    cls_name = getattr(cls, "__name__", cls)
    # Non-EXPANSIBLE classes can be found in configs
    cls_name_wo_config = cls_name.split(".")[0]
    unknown_fields = (
        set(json_obj)
        - {f[0] for f in cls.__annotations__.items()}
        - {cls_name_wo_config}
    )
    if unknown_fields:
        cls_fields = {f[0] for f in cls.__annotations__.items()}
        raise ConfigParseError(
            f"Unknown fields for class {cls_name} with fields {cls_fields} \
            detected in config json: {unknown_fields}"
        )
    for field, f_cls in cls.__annotations__.items():
        value = None
        is_optional = _is_optional(f_cls)

        if field not in json_obj:
            if field in cls._field_defaults:
                # if using default value, no conversion is needed
                value = cls._field_defaults.get(field)
        else:
            try:
                value = _value_from_json(f_cls, json_obj[field])
            except ConfigParseError:
                raise
            except Exception as e:
                raise ConfigParseError(
                    f"failed to parse {field} to {f_cls} with json payload \
                    {json_obj[field]}"
                ) from e
        # validate value
        if value is None and not is_optional:
            raise MissingValueError(
                f"missing value for {field} in class {cls_name} with json {json_obj}"
            )
        parsed_dict[field] = value

    return cls(**parsed_dict)


def _value_to_json(cls, value):
    cls_type = _get_class_type(cls)
    assert _is_optional(cls) or value is not None
    if value is None:
        return value
    # optional with more than 2 classes is treated as Union
    elif _is_optional(cls) and len(cls.__args__) == 2:
        sub_cls = cls.__args__[0] if type(None) != cls.__args__[0] else cls.__args__[1]
        return _value_to_json(sub_cls, value)
    elif cls_type == Union or getattr(cls, "__EXPANSIBLE__", False):
        real_cls = type(value)
        if hasattr(real_cls, "_fields"):
            value = config_to_json(real_cls, value)
        return {_canonical_typename(real_cls): value}
    elif cls_type == Any:
        return value
    # nested config
    elif hasattr(cls, "_fields"):
        return config_to_json(cls, value)
    elif issubclass(cls_type, Enum):
        return value.value
    elif issubclass(cls_type, List):
        sub_cls = cls.__args__[0]
        return [_value_to_json(sub_cls, v) for v in value]
    elif issubclass(cls_type, Tuple):
        return tuple(
            _value_to_json(c, v) for c, v in zip(_extend_tuple_type(cls, value), value)
        )
    elif issubclass(cls_type, Dict):
        sub_cls = cls.__args__[1]
        return {key: _value_to_json(sub_cls, v) for key, v in value.items()}
    return value


def config_to_json(cls, config_obj):
    json_result = {}
    if not hasattr(cls, "_fields"):
        raise IncorrectTypeError(f"{cls} is not a valid config class")
    for field, f_cls in cls.__annotations__.items():
        value = getattr(config_obj, field)
        json_result[field] = _value_to_json(f_cls, value)
    return json_result


def _get_class_type(cls):
    """
    type(cls) has an inconsistent behavior between 3.6 and 3.7 because of
    changes in the typing module. We therefore rely on __extra (3.6) and
    __origin__ (3.7), present only in classes from typing to extract the origin
    of the class for comparison, otherwise default to the type sent directly
    :param cls: class to infer
    :return: class or in the case of classes from typing module, the real type
    (Union, List) of the created object
    """
    return getattr(cls, "__extra__", getattr(cls, "__origin__", cls))


def _is_dict(obj):
    """support all dict-like types
    """
    return hasattr(obj, "__contains__") and hasattr(obj, "items")


def parse_config(config_json):
    """
    Parse PyTextConfig object from parameter string or parameter file
    """
    if "config" in config_json:
        config_json = config_json["config"]
    return pytext_config_from_json(config_json)

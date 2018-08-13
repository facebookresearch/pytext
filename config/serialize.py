#!/usr/bin/env python3
from enum import Enum
from typing import List, Union


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


def _union_from_json(union_cls, json_obj):
    if type(json_obj) is not dict:
        raise IncorrectTypeError(f"incorrect Union value {json_obj} for {union_cls}")
    type_name = list(json_obj)[0]

    for subclass in union_cls.__args__:
        if type(None) != subclass and type_name.lower() == subclass.__name__.lower():
            if hasattr(subclass, "_fields"):
                return config_from_json(subclass, json_obj[type_name])
            else:
                return json_obj[type_name]
    raise UnionTypeError(f"no suitable type found for {type_name} in union {union_cls}")


def _is_optional(cls):
    return type(cls) is type(Union) and type(None) in cls.__args__


def _enum_from_json(enum_cls, json_obj):
    for e in enum_cls:
        if e.value == json_obj:
            return e
    raise EnumTypeError("invalid enum value {json_obj} for {enum_cls}")


def _value_from_json(cls, value):
    if value is None:
        return value
    # optional with more than 2 classes is treated as Union
    if _is_optional(cls) and len(cls.__args__) == 2:
        sub_cls = cls.__args__[0] if type(None) != cls.__args__[0] else cls.__args__[1]
        value = config_from_json(sub_cls, value)
    # nested config
    elif hasattr(cls, "_fields"):
        value = config_from_json(cls, value)
    elif type(cls) is type(Union):
        value = _union_from_json(cls, value)
    elif issubclass(cls, Enum):
        value = _enum_from_json(cls, value)
    elif issubclass(cls, List):
        sub_cls = cls.__args__[0]
        value = [_value_from_json(sub_cls, v) for v in value]
    elif type(value) is not cls:
        raise IncorrectTypeError(
            f"incorrect type {type(value)} of value {value} for type {cls}"
        )
    return value


def config_from_json(cls, json_obj):
    parsed_dict = {}
    if not hasattr(cls, "_fields"):
        raise IncorrectTypeError(f"{cls} is not a valid config class")
    for field, f_cls in cls.__annotations__.items():
        value = None
        is_optional = _is_optional(f_cls)
        try:
            if field not in json_obj:
                if field in cls._field_defaults:
                    # if using default value, no conversion is needed
                    value = cls._field_defaults.get(field)
            else:
                value = _value_from_json(f_cls, json_obj[field])
            # validate value
            if value is None and not is_optional:
                raise MissingValueError(f"missing value for {field} in class {cls}")
            parsed_dict[field] = value
        except ConfigParseError as e:
            raise e
        except Exception as e:
            raise ConfigParseError(
                f"failed to parse {field} to {f_cls} with json payload \
                {value}"
            ) from e
    return cls(**parsed_dict)


def _value_to_json(cls, value):
    assert _is_optional(cls) or value is not None
    if value is None:
        return value
    # optional with more than 2 classes is treated as Union
    if _is_optional(cls) and len(cls.__args__) == 2:
        sub_cls = cls.__args__[0] if type(None) != cls.__args__[0] else cls.__args__[1]
        value = config_to_json(sub_cls, value)
    # nested config
    elif hasattr(cls, "_fields"):
        value = config_to_json(cls, value)
    elif type(cls) is type(Union):
        union_cls = type(value)
        if hasattr(union_cls, "_fields"):
            value = config_to_json(union_cls, value)
        value = {union_cls.__name__: value}
    elif issubclass(cls, Enum):
        value = value.value
    elif issubclass(cls, List):
        sub_cls = cls.__args__[0]
        value = [_value_to_json(sub_cls, v) for v in value]
    return value


def config_to_json(cls, config_obj):
    json_result = {}
    if not hasattr(cls, "_fields"):
        raise IncorrectTypeError(f"{cls} is not a valid config class")
    for field, f_cls in cls.__annotations__.items():
        value = getattr(config_obj, field)
        json_result[field] = _value_to_json(f_cls, value)
    return json_result

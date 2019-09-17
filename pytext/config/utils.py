#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Dict, List, Union


def is_component_class(obj):
    first = obj.__class__.__name__[0]
    return first.upper() == first


def find_param(root, suffix, parent=""):
    """
        Recursively look at all fields in config to find where `suffix` would fit.
        This is used to change configs so that they don't use default values.
        Return the list of field paths matching.
    """
    ret = []
    for k in getattr(root.__class__, "__annotations__", []):
        here = parent + k
        if here.endswith(suffix):
            ret += [here]

        v = getattr(root, k)
        if v is not None and is_component_class(type(v)):
            ret += find_param(v, suffix, parent=here + ".")

    return ret


def resolve_optional(type_v):
    """Deal with Optional implemented as Union[type, None]"""
    if getattr(type_v, "__origin__", None) == Union and len(type_v.__args__) == 2:
        if type_v.__args__[0] != type(None):
            return type_v.__args__[0]
        return type_v.__args__[1]
    return type_v


def cast_str(to_type, value):
    if type(value) != str:
        return value
    if to_type == int:
        return int(value)
    elif to_type == float:
        return float(value)
    elif to_type == str:
        return value
    elif to_type == bool:
        if value.lower() in ("yes", "true", "t", "1"):
            return True
        elif value.lower() in ("no", "false", "f", "0", ""):
            return False
        else:
            raise Exception(f'Not a boolean value: "{value}"')
    elif getattr(to_type, "__origin__", None) in (list, List):
        return [cast_str(to_type.__args__[0], v.strip()) for v in value.split(",")]
    elif getattr(to_type, "__origin__", None) in (dict, Dict):
        key_type, value_type = to_type.__args__
        ret = {}
        for entry in value.split(","):
            k, v = entry.split(":")
            typed_k = cast_str(key_type, k)
            typed_v = cast_str(value_type, v)
            ret[typed_k] = typed_v
        return ret
    else:
        raise Exception(f"Unsupported type: {to_type}")


def replace_param(root, path_list, value):
    for here in path_list[:-1]:
        root = getattr(root, here)

    param_name = path_list[-1]
    annotation = root.__class__.__annotations__[param_name]
    type_root = resolve_optional(annotation)
    setattr(root, param_name, cast_str(type_root, value))

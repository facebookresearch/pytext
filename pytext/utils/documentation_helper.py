#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from enum import Enum
from inspect import getmembers, isclass, isfunction
from sys import modules, stderr
from typing import Union

from pytext.config.component import get_component_name
from pytext.models.module import Module


ROOT_CONFIG = "PyTextConfig"


def eprint(*args, **kwargs):
    print(file=stderr, *args, **kwargs)


def get_class_members_recursive(obj):
    """Find all the field names for a given class and their default value."""
    ret = dict(vars(obj.Config if "Config" in dir(obj) else obj))
    for b in getattr(obj, "__bases__", []):
        # Only pytext configs.
        # Exclude Module because it adds load_path and save_path.
        if b.__module__.startswith("pytext") and b is not Module:
            for k, v in get_class_members_recursive(b).items():
                # Add missing members, don't override subclass with super
                if k not in ret:
                    ret[k] = v
    return ret


def get_config_fields(obj):
    """
        Return a dict of config help for this object, where:
        - key: config name
        - value: (default, type, options)
            - default: default value for this key if not specified
            - type: type for this config value, as a string
            - options: possible values for this config, only if type = Union

        If the type is "Union", the options give the lists of class names that
        are possible, and the default is one of those class names.
    """
    ret = {}
    members = get_class_members_recursive(obj)
    # Add fields with type by no default value
    typed_members = [k for k in members.get("__annotations__", {})]
    for k in typed_members:
        if k not in members:
            members[k] = None
    if issubclass(obj, Enum):
        opt = members["_member_names_"]
        typing = obj.__name__
        ret[typing] = (opt[0], typing, set(opt))
        return ret

    for k, v in sorted(members.items()):
        if k.startswith("_"):
            continue
        typing = members.get("__annotations__", {}).get(k)
        if issubclass(type(v), Enum):
            ret[k] = (v._name_, typing, set(type(v)._member_names_))
        elif not typing:
            if not isfunction(v):
                ret[k] = (v, None, None)
        # Done this way because type(Union) changed from Py3.6 to 3.7
        elif hasattr(typing, "__origin__") and typing.__origin__ == Union:
            options = set()
            for t in typing.__args__:
                options.add(get_component_name(t))
            ret[k] = (v, None, options)
        else:
            ret[k] = (v, get_component_name(typing), None)
    return ret


def pretty_print_config_class(obj):
    """Pretty-print the fields of one object."""
    parent_class_name = ""
    if hasattr(obj, "__bases__"):
        parent_classes = (b.__name__ for b in obj.__bases__)
        parent_class_name = ", ".join(parent_classes)
        print(f"=== {obj.__module__}.{obj.__name__} ({parent_class_name}) ===")
    else:
        print(f"=== {obj.__module__}.{obj.__name__} ===")
    if obj.__doc__:
        print(f'"""{obj.__doc__.strip()}"""')

    config_help = get_config_fields(obj)
    if issubclass(obj, Enum):
        for k, v in config_help.items():
            default, typing, options = v
            print(f"    {k}: ({typing})")
            for o in options:
                print(f"         {o}")
        return

    for k, v in config_help.items():
        default, typing, options = v
        if hasattr(default, "__module__"):
            default_value = get_component_name(default)
        else:
            default_value = default

        if typing and options:  # Enum
            print(f"    {k}: ({typing.__name__})")
            for o in options:
                if o == default_value:
                    print(f"         {o} (default)")
                else:
                    print(f"         {o}")
        elif options:  # Union
            print(f"    {k}: (one of)")
            for o in options:
                if o == default_value:
                    print(f"         {o} (default)")
                else:
                    print(f"         {o}")
        elif default and typing:
            print(f"    {k}: {typing} = {default_value}")
        elif default:
            print(f"    {k} = {default_value}")
        elif typing:
            print(f"    {k}: {typing}")
        else:
            print(f"    {k} = null")


def find_config_class(class_name):
    """
        Return the set of PyText classes matching that name.
        Handles fully-qualified `class_name` including module.
    """
    module_part = None
    if "." in class_name:
        m = class_name.split(".")
        class_name = m[-1]
        module_part = ".".join(m[:-1])

    ret = set()
    for mod_name, mod in modules.items():
        if not mod_name.startswith("pytext"):
            continue
        for name, obj in getmembers(mod, isclass):
            if name == class_name:
                if not module_part or obj.__module__ == module_part:
                    ret.add(obj)
    return ret

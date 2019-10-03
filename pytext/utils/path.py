#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os


def get_pytext_home():
    internal_home = os.path.realpath(os.path.join(__file__, "../../"))
    oss_home = os.path.realpath(os.path.join(__file__, "../../../"))
    default_home = ""
    if os.path.exists(os.path.join(internal_home, "tests")):
        default_home = internal_home
    elif os.path.exists(os.path.join(oss_home, "tests")):
        default_home = oss_home
    else:
        raise Exception("Can't find PYTEXT_HOME")
    pytext_home = os.environ.get("PYTEXT_HOME", default_home)
    print(f"PYTEXT_HOME: {pytext_home}")
    return pytext_home


PYTEXT_HOME = get_pytext_home()


def get_absolute_path(path):
    return (
        path
        if path.startswith("/")
        else os.path.realpath(os.path.join(PYTEXT_HOME, path))
    )

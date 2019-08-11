#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from sys import stderr


def eprint(*args, **kwargs):
    print(file=stderr, *args, **kwargs)

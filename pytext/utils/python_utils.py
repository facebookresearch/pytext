#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


def cls_vars(cls):
    return [v for n, v in vars(cls).items() if not n.startswith("_")]
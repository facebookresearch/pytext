#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch


def log_class_usage(klass):
    identifier = "PyText"
    if klass and hasattr(klass, "__name__"):
        identifier += f".{klass.__name__}"
    torch._C._log_api_usage_once(identifier)

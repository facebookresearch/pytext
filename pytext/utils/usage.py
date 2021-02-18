#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch

subsystem_name = "PyText"

def log_class_usage(klass):
    identifier = subsystem_name
    if klass and hasattr(klass, "__name__"):
        identifier += f".{klass.__name__}"
    torch._C._log_api_usage_once(identifier)


def log_feature_usage(feature):
    identifier = subsystem_name + "." + feature
    torch._C._log_api_usage_once(identifier)


def log_accelerator_feature_usage(feature):
    feature = "Accelerator." + feature
    log_feature_usage(feature)

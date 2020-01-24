#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Optional

import torch
from torch import nn


def prepare_full_key(instance_id: str, key: str, secondary_key: Optional[str] = None):
    if secondary_key is not None:
        return instance_id + "." + key + "." + secondary_key
    else:
        return instance_id + "." + key


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m

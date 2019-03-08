#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch

from .cuda import Variable, zerovar


def to_onehot(feat: Variable, size: int) -> Variable:
    """
    Transform features into one-hot vectors
    """
    dim = [d for d in feat.size()]
    vec_ = torch.unsqueeze(feat, len(dim))
    dim.append(size)
    one_hot = zerovar(dim)
    one_hot.data.scatter_(len(dim) - 1, vec_.data, 1)
    return one_hot

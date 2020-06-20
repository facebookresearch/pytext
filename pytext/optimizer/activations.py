#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch.nn as nn
from pytext.config.module_config import Activation


def get_activation(name, dim=1):
    if name == Activation.RELU:
        return nn.ReLU()
    elif name == Activation.LEAKYRELU:
        return nn.LeakyReLU()
    elif name == Activation.TANH:
        return nn.Tanh()
    elif name == Activation.GELU:
        return nn.GELU()
    elif name == Activation.GLU:
        return nn.GLU(dim=dim)
    else:
        raise RuntimeError(f"{name} is not supported")

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import random

import numpy as np
import torch


def cls_vars(cls):
    return [v for n, v in vars(cls).items() if not n.startswith("_")]


def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

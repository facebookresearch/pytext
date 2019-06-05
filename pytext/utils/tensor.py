#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List

import torch


@torch.jit.script
def xaviervar(size: List[int], device: str):
    t = torch.empty(size, device=device)
    t = torch.nn.init.xavier_normal_(t)
    return t

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.nn as nn


class ModelTransform(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        raise NotImplementedError

    def transform(self):
        raise NotImplementedError

    def collate_fn(self):
        raise NotImplementedError

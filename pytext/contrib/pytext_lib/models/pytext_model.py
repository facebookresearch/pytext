#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.nn as nn


class PyTextModel(nn.Module):
    def forward(self):
        raise NotImplementedError()

    def get_loss(self):
        raise NotImplementedError()

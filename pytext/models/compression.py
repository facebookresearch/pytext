#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.module as module


class DecompLinear(module.Module):
    """
    Variant of `torch.nn.Linear` that decomposes the weight matrix into two learnable
    sub-weights. The original weight of shape (in_dim, out_dim) is split with a middle
    dimension mid_dim. As a result, the two sub-weights have shapes (in_dim, mid_dim)
    and (mid_dim, out_dim), respectively. Unlike `torch.nn.Linear`, bias parameters
    are included by default. The input and output behavior of this module is the same
    as `torch.nn.Linear`.

    Args:
        in_dim (int): size of each input sample
        mid_dim (int): size of the intermediate dimension
        out_dim (int): size of each output sample

    """

    __constants__ = ["b1", "b2"]

    def __init__(self, in_dim, mid_dim, out_dim):
        super(DecompLinear, self).__init__()
        self.in_dim = in_dim
        self.mid_dim = mid_dim
        self.out_dim = out_dim
        self.w_a = nn.Parameter(torch.Tensor(self.out_dim, self.mid_dim))
        self.w_b = nn.Parameter(torch.Tensor(self.in_dim, self.mid_dim))
        self.b1 = nn.Parameter(torch.Tensor(self.in_dim))
        self.b2 = nn.Parameter(torch.Tensor(self.out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        # init weights
        nn.init.kaiming_uniform_(self.w_a, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w_b, a=math.sqrt(5))
        # init b1
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w_b)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.b1, -bound, bound)
        # init b2
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w_a)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.b2, -bound, bound)

    def forward(self, input):
        return F.linear(input, F.linear(self.w_a, self.w_b, self.b1), self.b2)

    def extra_repr(self):
        return "in_dim={}, mid_dim={}, out_dim={}".format(
            self.in_dim, self.mid_dim, self.out_dim
        )

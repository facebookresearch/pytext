#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List

from torch import nn
from torch.nn import functional as F


class GeLU(nn.Module):
    """Component class to wrap F.gelu."""

    def forward(self, input):
        return F.gelu(input.float()).type_as(input)


class ResidualMLP(nn.Module):
    """A square MLP component which can learn a bias on an input vector.
    This MLP in particular defaults to using GeLU as its activation function
    (this can be changed by passing a different activation function),
    and retains a residual connection to its original input to help with gradient
    propogation.

    Unlike pytext's MLPDecoder it doesn't currently allow adding a LayerNorm
    in between hidden layers.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.1,
        activation=GeLU,
    ):
        super().__init__()
        modules = []
        for last_dim, dim in zip([input_dim] + hidden_dims, hidden_dims):
            modules.extend(
                [nn.Linear(last_dim, dim), activation(), nn.Dropout(dropout)]
            )

        last_dim = hidden_dims[-1] if hidden_dims else input_dim
        # Unlike normal PyText mlp, we don't put an activation layer at the end.
        modules.extend([nn.Linear(last_dim, input_dim), nn.Dropout(dropout)])

        self.mlp = nn.Sequential(*modules)

    def forward(self, input):
        bias = self.mlp(input)
        return input + bias

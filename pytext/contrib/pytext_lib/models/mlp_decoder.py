#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List, Optional

import torch
import torch.nn as nn
from pytext.config.module_config import Activation
from pytext.optimizer import get_activation


class MLPDecoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool,
        hidden_dims: List[int] = None,
        activation: Activation = Activation.RELU,
    ) -> None:
        super().__init__()
        layers = []
        for dim in hidden_dims or []:
            layers.append(nn.Linear(in_dim, dim, bias))
            layers.append(get_activation(activation))
            in_dim = dim
        layers.append(nn.Linear(in_dim, out_dim, bias))

        self.mlp = nn.Sequential(*layers)

    def forward(
        self, representation: torch.Tensor, dense: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if dense is not None:
            representation = torch.cat([representation, dense], 1)
        return self.mlp(representation)

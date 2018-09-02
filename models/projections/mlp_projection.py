#!/usr/bin/env python3

from typing import List

import torch
import torch.nn as nn
from pytext.config import ConfigBase
from .projection_base import ProjectionBase


class MLPProjection(ProjectionBase):
    class Config(ConfigBase):
        # Intermediate hidden dimensions
        hidden_dims: List[int] = []

    def __init__(self, config: Config, from_dim: int, to_dim: int) -> None:
        super().__init__(config)
        layers = []
        for dim in config.hidden_dims or []:
            layers.append(nn.Linear(from_dim, dim))
            layers.append(nn.ReLU())
            from_dim = dim
        layers.append(nn.Linear(from_dim, to_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.mlp(x)

    def get_projection(self) -> List[nn.Module]:
        return self.mlp

#!/usr/bin/env python3

import torch
import torch.nn as nn
from typing import List
from .projection_base import ProjectionBase


class MLPProjection(ProjectionBase):
    def __init__(self, from_dim: int, hidden_dims: List[int], to_dim: int) -> None:
        super().__init__()
        layers = []
        for dim in hidden_dims:
            layers.append(nn.Linear(from_dim, dim))
            layers.append(nn.ReLU())
            from_dim = dim
        layers.append(nn.Linear(from_dim, to_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return [self.mlp(x)]

    def get_projection(self) -> List[nn.Module]:
        return [self.mlp]

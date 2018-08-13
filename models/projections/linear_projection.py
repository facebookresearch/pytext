#!/usr/bin/env python3

import torch
import torch.nn as nn
from typing import List
from .projection_base import ProjectionBase


class LinearProjection(ProjectionBase):
    def __init__(self, from_dim: int, to_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(from_dim, to_dim)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return [self.fc1(x)]

    def get_projection(self) -> List[nn.Module]:
        return [self.fc1]

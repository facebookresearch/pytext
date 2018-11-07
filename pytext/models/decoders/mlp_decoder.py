#!/usr/bin/env python3

from typing import List

import torch
import torch.nn as nn
from pytext.config import ConfigBase

from .decoder_base import DecoderBase


class MLPDecoder(DecoderBase):
    class Config(DecoderBase.Config, ConfigBase):
        # Intermediate hidden dimensions
        hidden_dims: List[int] = []

    def __init__(self, config: Config, from_dim: int, to_dim: int = 0) -> None:
        super().__init__(config)
        layers = []
        for dim in config.hidden_dims or []:
            layers.append(nn.Linear(from_dim, dim))
            layers.append(nn.ReLU())
            from_dim = dim
        if to_dim > 0:
            layers.append(nn.Linear(from_dim, to_dim))
        self.mlp = nn.Sequential(*layers)
        self.out_dim = to_dim if to_dim > 0 else config.hidden_dims[-1]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.mlp(x)

    def get_decoder(self) -> List[nn.Module]:
        return self.mlp

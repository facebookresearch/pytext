#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List

import torch
import torch.nn as nn
from pytext.config.module_config import Activation
from pytext.models.decoders.decoder_base import DecoderBase
from pytext.optimizer import get_activation
from pytext.utils import precision
from pytext.utils.usage import log_class_usage


class MLPDecoderTwoTower(DecoderBase):
    """
    Implements a 'two-tower' MLPDecoder: one for left and one for right
    """

    class Config(DecoderBase.Config):
        # Intermediate hidden dimensions
        right_hidden_dims: List[int] = []
        left_hidden_dims: List[int] = []
        hidden_dims: List[int] = []
        layer_norm: bool = False
        dropout: float = 0.0

    def __init__(
        self, config: Config, right_dim: int, left_dim: int, to_dim: int
    ) -> None:
        super().__init__(config)

        self.mlp_for_right = MLPDecoderTwoTower.get_mlp(
            right_dim,
            0,
            config.right_hidden_dims,
            config.layer_norm,
            config.dropout,
            export_embedding=True,
        )
        self.mlp_for_left = MLPDecoderTwoTower.get_mlp(
            left_dim,
            0,
            config.left_hidden_dims,
            config.layer_norm,
            config.dropout,
            export_embedding=True,
        )
        from_dim = config.right_hidden_dims[-1] + config.left_hidden_dims[-1]
        self.mlp = MLPDecoderTwoTower.get_mlp(
            from_dim, to_dim, config.hidden_dims, config.layer_norm, config.dropout
        )
        self.out_dim = to_dim
        log_class_usage

    @staticmethod
    def get_mlp(
        from_dim: int,
        to_dim: int,
        hidden_dims: List[int],
        layer_norm: bool,
        dropout: float,
        export_embedding: bool = False,
    ):
        layers = []
        for i in range(len(hidden_dims)):
            dim = hidden_dims[i]
            layers.append(nn.Linear(from_dim, dim, True))
            # Skip ReLU, LayerNorm, and dropout for the last layer if export_embedding
            if not (export_embedding and i == len(hidden_dims) - 1):
                layers.append(get_activation(Activation.RELU))
                if layer_norm:
                    layers.append(nn.LayerNorm(dim))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            from_dim = dim

        if to_dim > 0:
            layers.append(nn.Linear(from_dim, to_dim, True))

        return nn.Sequential(*layers)

    def forward(self, *x: List[torch.Tensor]) -> torch.Tensor:
        # x[0]: tokens, x[1]: right_dense, x[2]: left_dense
        assert len(x) == 3

        right_tensor = (
            torch.cat((x[0], x[1]), 1).half()
            if precision.FP16_ENABLED
            else torch.cat((x[0], x[1]), 1).float()
        )
        right_output = self.mlp_for_right(right_tensor)

        left_tensor = x[2].half() if precision.FP16_ENABLED else x[2].float()
        left_output = self.mlp_for_left(left_tensor)

        return self.mlp(torch.cat((right_output, left_output), 1))

    def get_decoder(self) -> List[nn.Module]:
        return [self.mlp_for_left, self.mlp_for_right]

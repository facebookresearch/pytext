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


# Export types are now ints
# -1 represents ExportType.None
# eg: to export from tower 0, set your export type to 0
#


class MLPDecoderNTower(DecoderBase):
    """
    Implements an 'n-tower' MLPDecoder
    """

    class Config(DecoderBase.Config):
        # Intermediate hidden dimensions
        tower_specific_hidden_dims: List[List[int]] = []
        hidden_dims: List[int] = []
        layer_norm: bool = False
        dropout: float = 0.0
        activation: Activation = Activation.RELU

    def __init__(
        self,
        config: Config,
        tower_dims: List[int],
        to_dim: int,
        export_type=-1,
    ) -> None:
        super().__init__(config)
        for i in range(len(tower_dims)):
            setattr(
                self,
                f"tower_mlp_{i}",
                MLPDecoderNTower.get_mlp(
                    tower_dims[i],
                    0,
                    config.tower_specific_hidden_dims[i],
                    config.layer_norm,
                    config.dropout,
                    config.activation,
                    export_embedding=True,
                ),
            )
        from_dim = 0
        for dims in config.tower_specific_hidden_dims:
            from_dim += dims[-1]

        self.mlp = MLPDecoderNTower.get_mlp(
            from_dim,
            to_dim,
            config.hidden_dims,
            config.layer_norm,
            config.dropout,
            config.activation,
        )
        self.out_dim = to_dim
        self.export_type = export_type
        log_class_usage

    @staticmethod
    def get_mlp(
        from_dim: int,
        to_dim: int,
        hidden_dims: List[int],
        layer_norm: bool,
        dropout: float,
        activation: Activation,
        export_embedding: bool = False,
    ):
        layers = []
        for i in range(len(hidden_dims)):
            dim = hidden_dims[i]
            layers.append(nn.Linear(from_dim, dim, True))
            # Skip ReLU, LayerNorm, and dropout for the last layer if export_embedding
            if not (export_embedding and i == len(hidden_dims) - 1):
                layers.append(get_activation(activation))
                if layer_norm:
                    layers.append(nn.LayerNorm(dim))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            from_dim = dim

        if to_dim > 0:
            layers.append(nn.Linear(from_dim, to_dim, True))

        return nn.Sequential(*layers)

    def forward(self, *x: List[torch.Tensor]) -> torch.Tensor:
        # as per the associated model's arrange_model_inputs()
        # first half of the list is the token inputs, the second half is the dense features
        halfway = len(x) // 2
        outputs = []

        for i in range(halfway):
            if self.export_type == i or self.export_type == -1:
                tensor = (
                    torch.cat((x[i], x[halfway + i]), 1).half()
                    if precision.FP16_ENABLED
                    else torch.cat((x[i], x[halfway + i]), 1).float()
                )
                # len(tensor i) == i's encoder.embedding_dim + i's dense_dim
                output = getattr(self, f"tower_mlp_{i}")(tensor)
                outputs.append(output)
                if self.export_type == i:
                    return output

        return self.mlp(torch.cat(outputs, 1))

    def get_decoder(self) -> List[nn.Module]:
        return [
            getattr(self, f"tower_mlp_{i}")
            for i in range(len(self.tower_specific_hidden_dims))
        ]

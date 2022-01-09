#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from enum import Enum
from typing import List

import torch
import torch.nn as nn
from pytext.config.module_config import Activation
from pytext.models.decoders.decoder_base import DecoderBase
from pytext.optimizer import get_activation
from pytext.utils import precision
from pytext.utils.usage import log_class_usage


class ExportType(Enum):
    RIGHT = "RIGHT"
    MIDDLE = "MIDDLE"
    LEFT = "LEFT"
    NONE = "NONE"


class MLPDecoderTriTower(DecoderBase):
    """
    Implements a 'tri-tower' MLPDecoder: one for left, one for middle, and one for right
    """

    class Config(DecoderBase.Config):
        # Intermediate hidden dimensions
        right_hidden_dims: List[int] = []
        middle_hidden_dims: List[int] = []
        left_hidden_dims: List[int] = []
        hidden_dims: List[int] = []
        layer_norm: bool = False
        dropout: float = 0.0
        activation: Activation = Activation.RELU

    def __init__(
        self,
        config: Config,
        right_dim: int,
        middle_dim: int,
        left_dim: int,
        to_dim: int,
        export_type=ExportType.NONE,
    ) -> None:
        super().__init__(config)

        self.mlp_for_right = MLPDecoderTriTower.get_mlp(
            right_dim,
            0,
            config.right_hidden_dims,
            config.layer_norm,
            config.dropout,
            config.activation,
            export_embedding=True,
        )
        self.mlp_for_middle = MLPDecoderTriTower.get_mlp(
            middle_dim,
            0,
            config.middle_hidden_dims,
            config.layer_norm,
            config.dropout,
            config.activation,
            export_embedding=True,
        )
        self.mlp_for_left = MLPDecoderTriTower.get_mlp(
            left_dim,
            0,
            config.left_hidden_dims,
            config.layer_norm,
            config.dropout,
            config.activation,
            export_embedding=True,
        )
        from_dim = (
            config.right_hidden_dims[-1]
            + config.middle_hidden_dims[-1]
            + config.left_hidden_dims[-1]
        )
        self.mlp = MLPDecoderTriTower.get_mlp(
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
        # as per TriTowerClassificationModel's arrange_model_inputs()
        # x[0]: right_text_emb, x[1]: middle_text_emb, x[2]: left_text_emb, x[3]: right_dense, x[4]: middle_dense, x[5]: left_dense
        assert len(x) == 6

        if self.export_type == ExportType.RIGHT or self.export_type == ExportType.NONE:
            right_tensor = (
                torch.cat((x[0], x[3]), 1).half()
                if precision.FP16_ENABLED
                else torch.cat((x[0], x[3]), 1).float()
            )
            # len(right_tensor[0]) == right_encoder.embedding_dim + right_dense_dim
            right_output = self.mlp_for_right(right_tensor)
            if self.export_type == ExportType.RIGHT:
                return right_output
        if self.export_type == ExportType.MIDDLE or self.export_type == ExportType.NONE:
            middle_tensor = (
                torch.cat((x[1], x[4]), 1).half()
                if precision.FP16_ENABLED
                else torch.cat((x[1], x[4]), 1).float()
            )
            # len(middle_tensor[0]) == middle_encoder.embedding_dim + middle_dense_dim
            middle_output = self.mlp_for_middle(middle_tensor)
            if self.export_type == ExportType.MIDDLE:
                return middle_output

        if self.export_type == ExportType.LEFT or self.export_type == ExportType.NONE:
            left_tensor = (
                torch.cat((x[2], x[5]), 1).half()
                if precision.FP16_ENABLED
                else torch.cat((x[2], x[5]), 1).float()
            )
            # len(left_tensor[0]) == left_encoder.embedding_dim + left_dense_dim
            left_output = self.mlp_for_left(left_tensor)
            if self.export_type == ExportType.LEFT:
                return left_output

        return self.mlp(torch.cat((right_output, middle_output, left_output), 1))

    def get_decoder(self) -> List[nn.Module]:
        return [self.mlp_for_left, self.mpl_for_middle, self.mlp_for_right]

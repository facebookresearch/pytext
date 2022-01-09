#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from enum import Enum
from typing import List, Optional

import torch
import torch.nn as nn
from pytext.config.module_config import Activation
from pytext.models.decoders.decoder_base import DecoderBase
from pytext.optimizer import get_activation
from pytext.utils import precision
from pytext.utils.file_io import PathManager
from pytext.utils.usage import log_class_usage
from torch.serialization import default_restore_location


class ExportType(Enum):
    RIGHT = "RIGHT"
    LEFT = "LEFT"
    NONE = "NONE"


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
        dense_tower_dims: Optional[List[int]] = None
        load_model_path: Optional[str] = None
        load_strict: Optional[bool] = False
        activation: Activation = Activation.RELU

    def __init__(
        self,
        config: Config,
        right_dim: int,
        left_dim: int,
        to_dim: int,
        export_type=ExportType.NONE,
    ) -> None:
        super().__init__(config)

        self.mlp_for_right = MLPDecoderTwoTower.get_mlp(
            right_dim,
            0,
            config.right_hidden_dims,
            config.layer_norm,
            config.dropout,
            config.activation,
            export_embedding=True,
        )
        self.mlp_for_left = MLPDecoderTwoTower.get_mlp(
            left_dim,
            0,
            config.left_hidden_dims,
            config.layer_norm,
            config.dropout,
            config.activation,
            export_embedding=True,
        )

        if config.dense_tower_dims is None:
            from_dim = config.right_hidden_dims[-1] + config.left_hidden_dims[-1]
            self.concat_dense = False
        else:
            self.mlp_for_dense = MLPDecoderTwoTower.get_mlp(
                from_dim=config.dense_tower_dims[0],
                to_dim=config.dense_tower_dims[-1],
                hidden_dims=config.dense_tower_dims[1:-1],
                layer_norm=config.layer_norm,
                dropout=config.dropout,
                activation=config.activation,
            )
            from_dim = (
                config.right_hidden_dims[-1]
                + config.left_hidden_dims[-1]
                + config.dense_tower_dims[-1]
            )
            self.concat_dense = True

        self.mlp = MLPDecoderTwoTower.get_mlp(
            from_dim,
            to_dim,
            config.hidden_dims,
            config.layer_norm,
            config.dropout,
            config.activation,
        )

        # load model
        if config.load_model_path:
            with PathManager.open(config.load_model_path, "rb") as f:
                model = torch.load(
                    f, map_location=lambda s, l: default_restore_location(s, "cpu")
                )
            mlp_state = {
                k.replace("decoder.", ""): v
                for k, v in model["model_state"].items()
                if k.startswith("decoder.mlp")
                or k.startswith("decoder.mlp_for_right")
                or k.startswith("decoder.mlp_for_left")
                or k.startswith("decoder.mlp_for_dense")
            }
            if mlp_state["mlp.0.weight"].shape[1] != from_dim:
                mlp_state = {
                    k: v for k, v in mlp_state.items() if not k.startswith("mlp.")
                }
                print("top mlp weights not loaded")
            self.load_state_dict(mlp_state, strict=config.load_strict)
            print("loaded mlp state")

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
        # x[0]: right_text_emb, x[1]: left_text_emb, x[2]: right_dense, x[3]: left_dense

        if self.concat_dense:
            assert len(x) == 5
        else:
            assert len(x) == 4

        if self.export_type == ExportType.RIGHT or self.export_type == ExportType.NONE:
            right_tensor = (
                torch.cat((x[0], x[2]), 1).half()
                if precision.FP16_ENABLED
                else torch.cat((x[0], x[2]), 1).float()
            )
            right_output = self.mlp_for_right(right_tensor)
            if self.export_type == ExportType.RIGHT:
                return right_output

        if self.export_type == ExportType.LEFT or self.export_type == ExportType.NONE:
            left_tensor = (
                torch.cat((x[1], x[3]), 1).half()
                if precision.FP16_ENABLED
                else torch.cat((x[1], x[3]), 1).float()
            )
            left_output = self.mlp_for_left(left_tensor)
            if self.export_type == ExportType.LEFT:
                return left_output

        if not self.concat_dense:
            return self.mlp(torch.cat((right_output, left_output), 1))
        else:
            if self.export_type == ExportType.NONE:
                dense_tensor = x[4].half() if precision.FP16_ENABLED else x[4].float()
            else:
                dense_tensor = x[4]

            dense = self.mlp_for_dense(dense_tensor)
            return self.mlp(torch.cat((right_output, left_output, dense), 1))

    def get_decoder(self) -> List[nn.Module]:
        return [self.mlp_for_left, self.mlp_for_right]

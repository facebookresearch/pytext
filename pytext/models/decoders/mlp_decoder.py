#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List, Optional

import torch
import torch.nn as nn
from pytext.config.module_config import Activation
from pytext.optimizer import get_activation
from pytext.utils.usage import log_class_usage

from .decoder_base import DecoderBase


class MLPDecoder(DecoderBase):
    """
    `MLPDecoder` implements a fully connected network and uses ReLU as the
    activation function. The module projects an input tensor to `out_dim`.

    Args:
        config (Config): Configuration object of type MLPDecoder.Config.
        in_dim (int): Dimension of input Tensor passed to MLP.
        out_dim (int): Dimension of output Tensor produced by MLP. Defaults to 0.

    Attributes:
        mlp (type): Module that implements the MLP.
        out_dim (type): Dimension of the output of this module.
        hidden_dims (List[int]): Dimensions of the outputs of hidden layers.

    """

    class Config(DecoderBase.Config):
        """
        Configuration class for `MLPDecoder`.

        Attributes:
            hidden_dims (List[int]): Dimensions of the outputs of hidden layers..
            temperature (float): Scales logits by this value (before the softmax
            operation) during test-time only. Temperature scaling has no effect on
            the top prediction but changes the shape of the posterior distribution,
            which can be useful for a range of tasks (e.g., model calibration).
        """

        hidden_dims: List[int] = []
        out_dim: Optional[int] = None
        layer_norm: bool = False
        dropout: float = 0.0
        bias: bool = True
        activation: Activation = Activation.RELU
        temperature: float = 1.0
        spectral_normalization: bool = False

    def __init__(self, config: Config, in_dim: int, out_dim: int = 0) -> None:
        super().__init__(config)

        layers = []
        for dim in config.hidden_dims or []:
            layers.append(nn.Linear(in_dim, dim, config.bias))
            layers.append(get_activation(config.activation))
            if config.layer_norm:
                layers.append(nn.LayerNorm(dim))
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            in_dim = dim
        if config.out_dim is not None:
            out_dim = config.out_dim
        if out_dim > 0:
            layers.append(nn.Linear(in_dim, out_dim, config.bias))

        assert len(layers) > 0
        if config.spectral_normalization:
            layers[-1] = torch.nn.utils.spectral_norm(layers[-1])
        self.mlp = nn.Sequential(*layers)
        self.out_dim = out_dim if out_dim > 0 else config.hidden_dims[-1]
        self.temperature = config.temperature
        log_class_usage(__class__)

    def forward(self, *input: torch.Tensor) -> torch.Tensor:
        mlp_out = self.mlp(torch.cat(input, 1))
        return (
            mlp_out
            if self.training or self.temperature == 1.0
            else mlp_out / self.temperature
        )

    def get_decoder(self) -> List[nn.Module]:
        """Returns the MLP module that is used as a decoder."""
        return [self.mlp]

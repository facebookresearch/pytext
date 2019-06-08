#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List, Optional

import torch
import torch.nn as nn

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
        """

        hidden_dims: List[int] = []
        out_dim: Optional[int] = None
        layer_norm: bool = False

    def __init__(self, config: Config, in_dim: int, out_dim: int = 0) -> None:
        super().__init__(config)

        layers = []
        for dim in config.hidden_dims or []:
            layers.append(nn.Linear(in_dim, dim))
            layers.append(nn.ReLU())
            if config.layer_norm:
                layers.append(nn.LayerNorm(dim))
            in_dim = dim
        if config.out_dim:
            out_dim = config.out_dim
        if out_dim > 0:
            layers.append(nn.Linear(in_dim, out_dim))

        self.mlp = nn.Sequential(*layers)
        self.out_dim = out_dim if out_dim > 0 else config.hidden_dims[-1]

    def forward(self, *input: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat(input, 1))

    def get_decoder(self) -> List[nn.Module]:
        """Returns the MLP module that is used as a decoder.
        """
        return [self.mlp]

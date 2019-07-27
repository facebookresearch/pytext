#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math

import torch
import torch.nn as nn
from pytext.config.module_config import CNNParams
from pytext.models.representations.representation_base import RepresentationBase


class DeepCNNRepresentation(RepresentationBase):
    """
    `DeepCNNRepresentation` implements CNN representation layer
    preceded by a dropout layer. CNN representation layer is based on the encoder
    in the architecture proposed by Gehring et. al. in Convolutional Sequence to
    Sequence Learning.

    Args:
        config (Config): Configuration object of type DeepCNNRepresentation.Config.
        embed_dim (int): The number of expected features in the input.

    """

    class Config(RepresentationBase.Config):
        cnn: CNNParams = CNNParams()
        dropout: float = 0.3

    def __init__(self, config: Config, embed_dim: int) -> None:
        super().__init__(config)

        out_channels = config.cnn.kernel_num
        kernel_sizes = config.cnn.kernel_sizes
        weight_norm = config.cnn.weight_norm
        dilated = config.cnn.dilated

        conv_layers = []
        linear_layers = []
        in_channels = embed_dim

        for i, k in enumerate(kernel_sizes):
            assert (k - 1) % 2 == 0

            proj = (
                nn.Linear(in_channels, out_channels)
                if in_channels != out_channels
                else None
            )
            linear_layers.append(proj)

            dilation = 2 ** i if dilated else 1
            padding = (k - 1) // 2

            single_conv = nn.Conv1d(
                in_channels, 2 * out_channels, k, padding=padding, dilation=dilation
            )
            single_conv = (
                nn.utils.weight_norm(single_conv) if weight_norm else single_conv
            )
            conv_layers.append(single_conv)

            in_channels = out_channels

        self.convs = nn.ModuleList(conv_layers)
        self.projections = nn.ModuleList(linear_layers)
        self.glu = nn.GLU(dim=1)

        self.representation_dim = out_channels
        self.dropout = nn.Dropout(p=config.dropout)

    def forward(self, inputs: torch.Tensor, *args) -> torch.Tensor:
        inputs = self.dropout(inputs)
        # bsz * seq_len * embed_dim -> bsz * embed_dim * seq_len
        words = inputs.permute(0, 2, 1)
        for conv, proj in zip(self.convs, self.projections):
            if proj is None:
                residual = words
            else:
                tranposed = words.permute(0, 2, 1)
                residual = proj(tranposed).permute(0, 2, 1)
            words = conv(words)
            words = self.glu(words)
            words = (words + residual) * math.sqrt(0.5)
        return words.permute(0, 2, 1)

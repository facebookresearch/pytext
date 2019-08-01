#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math

import torch
import torch.nn as nn
from pytext.config.module_config import Activation, CNNParams
from pytext.models.representations.representation_base import RepresentationBase
from pytext.optimizer import get_activation


class Trim1d(nn.Module):
    """
    Trims a 1d convolutional output. Used to implement history-padding
    by removing excess padding from the right.

    """

    def __init__(self, trim):
        super(Trim1d, self).__init__()

        self.trim = trim

    def forward(self, x):
        return x[:, :, : -self.trim].contiguous()


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
        activation: Activation = Activation.GLU

    def __init__(self, config: Config, embed_dim: int) -> None:
        super().__init__(config)

        out_channels = config.cnn.kernel_num
        kernel_sizes = config.cnn.kernel_sizes
        weight_norm = config.cnn.weight_norm
        dilated = config.cnn.dilated
        causal = config.cnn.causal
        activation = config.activation

        conv_layers = []
        trim_layers = []
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
            padding = (k - 1) * dilation if causal else ((k - 1) // 2) * dilation

            single_conv = nn.Conv1d(
                in_channels,
                (out_channels * 2 if activation == Activation.GLU else out_channels),
                k,
                padding=padding,
                dilation=dilation,
            )
            single_conv = (
                nn.utils.weight_norm(single_conv) if weight_norm else single_conv
            )
            conv_layers.append(single_conv)

            # Non-causal convolutions are centered, so they will consume
            # ((k - 1) // 2) * d padding on both the left and the right of the sequence.
            # Causal convolutions are shifted to the left (to account for temporal
            # ordering), so they will only consume padding from the left. Therefore,
            # we pad this side with the full amount (k - 1) * d.
            trim = Trim1d(padding) if causal else None
            trim_layers.append(trim)

            in_channels = out_channels

        self.convs = nn.ModuleList(conv_layers)
        self.trims = nn.ModuleList(trim_layers)
        self.projections = nn.ModuleList(linear_layers)
        self.activation = get_activation(activation)

        self.representation_dim = out_channels
        self.dropout = nn.Dropout(p=config.dropout)

    def forward(self, inputs: torch.Tensor, *args) -> torch.Tensor:
        inputs = self.dropout(inputs)
        # bsz * seq_len * embed_dim -> bsz * embed_dim * seq_len
        words = inputs.permute(0, 2, 1)
        for conv, trim, proj in zip(self.convs, self.trims, self.projections):
            if proj:
                tranposed = words.permute(0, 2, 1)
                residual = proj(tranposed).permute(0, 2, 1)
            else:
                residual = words
            words = conv(words)
            if trim:
                words = trim(words)
            words = self.activation(words)
            words = (words + residual) * math.sqrt(0.5)
        return words.permute(0, 2, 1)

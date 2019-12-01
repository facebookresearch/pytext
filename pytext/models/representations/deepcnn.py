#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math

import torch
import torch.nn as nn
from pytext.config.module_config import Activation, CNNParams, PoolingType
from pytext.models.representations.representation_base import RepresentationBase
from pytext.optimizer import get_activation


def pool(pooling_type, words):
    # input dims: bsz * seq_len * num_filters
    if pooling_type == PoolingType.MEAN:
        return words.mean(dim=1)
    elif pooling_type == PoolingType.MAX:
        return words.max(dim=1)[0]
    elif pooling_type == PoolingType.NONE:
        return words
    else:
        return NotImplementedError


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


class SeparableConv1d(nn.Module):
    """
    Implements a 1d depthwise separable convolutional layer. In regular convolutional
    layers, the input channels are mixed with each other to produce each output channel.
    Depthwise separable convolutions decompose this process into two smaller
    convolutions -- a depthwise and pointwise convolution.

    The depthwise convolution spatially convolves each input channel separately,
    then the pointwise convolution projects this result into a new channel space.
    This process reduces the number of FLOPS used to compute a convolution and also
    exhibits a regularization effect. The general behavior -- including the input
    parameters -- is equivalent to `nn.Conv1d`.

    `bottleneck` controls the behavior of the pointwise convolution. Instead of
    upsampling directly, we split the pointwise convolution into two pieces: the first
    convolution downsamples into a (sufficiently small) low dimension and the
    second convolution upsamples into the target (higher) dimension. Creating this
    bottleneck significantly cuts the number of parameters with minimal loss
    in performance.

    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        padding: int,
        dilation: int,
        bottleneck: int,
    ):
        super(SeparableConv1d, self).__init__()

        conv_layers = [
            nn.Conv1d(
                input_channels,
                input_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
                groups=input_channels,
            )
        ]

        if bottleneck > 0:
            conv_layers.extend(
                [
                    nn.Conv1d(input_channels, bottleneck, 1),
                    nn.Conv1d(bottleneck, output_channels, 1),
                ]
            )
        else:
            conv_layers.append(nn.Conv1d(input_channels, output_channels, 1))

        self.conv = nn.Sequential(*conv_layers)

    def forward(self, x):
        return self.conv(x)


def create_conv_package(
    index: int,
    activation: Activation,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    causal: bool,
    dilated: bool,
    separable: bool,
    bottleneck: int,
    weight_norm: bool,
):
    """
    Creates a convolutional layer with the specified arguments.

    Args:
        index (int): Index of a convolutional layer in the stack.
        activation (Activation): Activation function.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of 1d convolutional filter.
        causal (bool): Whether the convolution is causal or not. If set, it
        accounts for the temporal ordering of the inputs.
        dilated (bool): Whether the convolution is dilated or not. If set,
        the receptive field of the convolutional stack grows exponentially.
        separable (bool): Whether to use depthwise separable convolutions
        or not -- see `SeparableConv1d`.
        bottleneck (int): Bottleneck channel dimension for depthwise separable
        convolutions. See `SeparableConv1d` for an in-depth explanation.
        weight_norm (bool): Whether to add weight normalization to the
        regular convolutions or not.

    """

    if not separable and bottleneck > 0:
        raise RuntimeError(
            "Bottleneck layers can only be used with separable convolutions"
        )

    if separable and weight_norm:
        raise RuntimeError(
            "Weight normalization is not supported for separable convolutions"
        )

    def _compute_dilation(index, dilated):
        """
        If set, the dilation factor increases by a factor of two for each
        successive convolution to increase the receptive field exponentially.

        """

        if dilated:
            return 2 ** index
        return 1

    def _compute_padding(kernel_size, dilation, causal):
        """
        Non-causal convolutions are centered, so they will consume ((k - 1) // 2) * d
        padding on both the left and the right of the sequence. Causal convolutions
        are shifted to the left (to account for temporal ordering), so they will
        only consume padding from the left. Therefore, we pad this side with the
        full amount (k - 1) * d and remove the excess right-padding with `Trim1d`.

        """

        if causal:
            return (kernel_size - 1) * dilation
        return ((kernel_size - 1) // 2) * dilation

    def _compute_out_channels(out_channels, activation):
        """
        Gated Linear Unit (GLU) activations train two groups of convolutions,
        then linearly combine their outputs through a gating mechanism. We
        double the number of `out_channels` to mimic these two groups.

        """

        if activation == Activation.GLU:
            return out_channels * 2
        return out_channels

    package = []
    dilation = _compute_dilation(index, dilated)
    padding = _compute_padding(kernel_size, dilation, causal)
    out_channels = _compute_out_channels(out_channels, activation)

    if separable:
        package.append(
            SeparableConv1d(
                in_channels, out_channels, kernel_size, padding, dilation, bottleneck
            )
        )
    else:
        conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=padding, dilation=dilation
        )
        if weight_norm:
            conv = nn.utils.weight_norm(conv)
        package.append(conv)

    if causal:
        package.append(Trim1d(padding))

    return package[0] if len(package) == 1 else nn.Sequential(*package)


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
        separable: bool = False
        bottleneck: int = 0
        pooling_type: PoolingType = PoolingType.NONE

    def __init__(self, config: Config, embed_dim: int) -> None:
        super().__init__(config)

        out_channels = config.cnn.kernel_num
        kernel_sizes = config.cnn.kernel_sizes
        weight_norm = config.cnn.weight_norm
        dilated = config.cnn.dilated
        causal = config.cnn.causal

        activation = config.activation
        pooling_type = config.pooling_type
        separable = config.separable
        bottleneck = config.bottleneck

        conv_layers = {}
        linear_layers = {}
        in_channels = embed_dim

        for i, k in enumerate(kernel_sizes):
            assert (k - 1) % 2 == 0

            if in_channels != out_channels:
                linear_layers[str(i)] = nn.Linear(in_channels, out_channels)

            single_conv = create_conv_package(
                index=i,
                activation=activation,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=k,
                causal=causal,
                dilated=dilated,
                separable=separable,
                bottleneck=bottleneck,
                weight_norm=weight_norm,
            )
            conv_layers[str(i)] = single_conv

            in_channels = out_channels

        self.convs = nn.ModuleDict(conv_layers)
        self.projections = nn.ModuleDict(linear_layers)
        self.activation = get_activation(activation)
        self.pooling_type = pooling_type

        self.representation_dim = out_channels
        self.dropout = nn.Dropout(p=config.dropout)

    def forward(self, inputs: torch.Tensor, *args) -> torch.Tensor:
        inputs = self.dropout(inputs)
        # bsz * seq_len * embed_dim -> bsz * embed_dim * seq_len
        words = inputs.permute(0, 2, 1)
        convs_keys = self.convs.keys()
        projections_keys = self.projections.keys()
        # Extra verbosity is due to jit.script.
        for k in convs_keys:
            conv = self.convs[k]
            if k not in projections_keys:
                residual = words
            else:
                proj = self.projections[k]
                tranposed = words.permute(0, 2, 1)
                residual = proj(tranposed).permute(0, 2, 1)
            words = conv(words)
            words = self.activation(words)
            words = (words + residual) * math.sqrt(0.5)
        words = words.permute(0, 2, 1)
        return pool(self.pooling_type, words)

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Dict, Optional

import torch.jit
import torch.nn as nn
from pytext.config import ConfigBase
from torch import Tensor

from .base import PyTextIncrementalDecoderComponent
from .utils import unfold1d


class LightweightConv(PyTextIncrementalDecoderComponent):
    class Config(ConfigBase):
        num_heads: int = 2
        weight_softmax: bool = False
        bias: bool = True

    @classmethod
    def from_config(cls, config, input_size, kernel_size, convolution_type):
        return cls(input_size, kernel_size, convolution_type, **config._asdict())

    def __init__(
        self,
        input_size,
        kernel_size,
        # ARBABU TODO : convert this to a enum
        convolution_type: str,
        num_heads,
        weight_softmax,
        bias,
    ):
        super().__init__()
        self.input_size = input_size
        self.kernel_size = kernel_size
        if convolution_type == "non-causal":
            padding_l = (
                kernel_size // 2
                if kernel_size % 2 == 1
                else ((kernel_size - 1) // 2, kernel_size // 2)
            )
        elif convolution_type == "causal":
            padding_l = kernel_size - 1
        else:
            raise Exception("Convolution type not supported")
        self.padding_l = padding_l
        self.num_heads = num_heads
        self.weight_softmax = weight_softmax

        self.weight = nn.Parameter(torch.Tensor(num_heads, 1, kernel_size))

        self.has_bias = bias
        if bias:
            self.bias = nn.Parameter(torch.Tensor(input_size).view(1, 1, -1))
        else:
            self.bias = nn.Parameter(torch.Tensor())

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.has_bias:
            nn.init.constant_(self.bias, 0.0)

    def forward(self, x, incremental_state: Optional[Dict[str, Tensor]] = None):
        """Assuming the input, x, of the shape T x B x C and producing an output in the shape T x B x C
        args:
            x: Input of shape T x B x C, i.e. (timesteps, batch_size, input_size)
            incremental_state: A dict to keep the state
        """

        output = self._forward_unfolded(x, incremental_state)

        if self.has_bias:
            output = output + self.bias
        return output

    def _forward_unfolded(
        self, x, incremental_state: Optional[Dict[str, Tensor]] = None
    ):
        """The conventional implementation of convolutions.
        Unfolding the input by having a window shifting to the right."""
        T, B, C = x.size()
        K, H = self.kernel_size, self.num_heads
        R = C // H
        assert R * H == C == self.input_size

        weight = self.weight.view(H, K)
        if incremental_state is not None:
            input_buffer = self._get_input_buffer(incremental_state)
            if input_buffer is not None:
                x_unfold = torch.cat([input_buffer, x.unsqueeze(3)], dim=3)
            else:
                # First decoder step
                x_unfold = x.unsqueeze(3).clone()
            if self.kernel_size > 1:
                self._set_input_buffer(
                    incremental_state, x_unfold[:, :, :, -self.kernel_size + 1 :]
                )
            x_unfold = x_unfold.view(T * B * H, R, -1)
        else:
            # unfold the input: T x B x C --> T' x B x C x K
            x_unfold = unfold1d(x, self.kernel_size, self.padding_l, 0.0)
            x_unfold = x_unfold.view(T * B * H, R, K)

        if incremental_state is not None:
            weight = weight[:, -(x_unfold.size(2)) :]
            K = weight.size(1)

        weight = (
            weight.view(1, H, K).expand(T * B, H, K).contiguous().view(T * B * H, K, 1)
        )

        output = torch.bmm(x_unfold, weight)  # T*B*H x R x 1
        output = output.view(T, B, C)
        return output

    def reorder_incremental_state(
        self, incremental_state: Dict[str, Tensor], new_order: Tensor
    ):
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            input_buffer = input_buffer.index_select(1, new_order)
            self._set_input_buffer(incremental_state, input_buffer)

    def _get_input_buffer(self, incremental_state: Dict[str, Tensor]):
        return self.get_incremental_state(incremental_state, "input_buffer")

    def _set_input_buffer(
        self, incremental_state: Dict[str, Tensor], new_buffer: Tensor
    ):
        return self.set_incremental_state(incremental_state, "input_buffer", new_buffer)

    def extra_repr(self):
        s = "{}, kernel_size={}, padding_l={}, num_heads={}, weight_softmax={}, bias={}".format(
            self.input_size,
            self.kernel_size,
            self.padding_l,
            self.num_heads,
            self.weight_softmax,
            self.has_bias,
        )
        return s

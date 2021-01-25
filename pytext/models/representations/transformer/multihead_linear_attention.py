#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Tuple

import torch
from pytext.utils.usage import log_class_usage
from torch import nn
from torch.nn import functional as F


class MultiheadLinearAttention(nn.Module):
    """
    This is a TorchScriptable implementation of MultiheadLinearAttention:
    https://arxiv.org/pdf/2006.04768.pdf. from fairseq for the purposes of
    creating a productionized Linformer model. It distills just
    the elements which are required to implement the RoBERTa use cases of
    MultiheadLinearAttention, and within that is restructured and rewritten to be able
    to be compiled by TorchScript for production use cases.

    The default constructor values match those required to import the public
    RoBERTa weights. Unless you are pretraining your own model, there's no need to
    change them.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        scaling: float = 0.125,
        dropout: float = 0.1,
        compress_layer=None,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = scaling
        self.dropout = nn.Dropout(dropout)
        self.kput_projection = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.vput_projection = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.qput_projection = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.output_projection = nn.Linear(embed_dim, embed_dim)
        self.compress_k = compress_layer
        log_class_usage(__class__)

    def get_compressed_projection(
        self, k_input: torch.Tensor, v_input: torch.Tensor, target_length: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        k_input = (
            F.linear(k_input, self.compress_k.weight[:, 0:target_length])
            .permute(2, 0, 1)
            .contiguous()
        )
        v_input = (
            F.linear(v_input, self.compress_k.weight[:, 0:target_length])
            .permute(2, 0, 1)
            .contiguous()
        )
        return k_input, v_input

    def forward(self, query, key_padding_mask):
        """Input shape: Time x Batch x Channel
        Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x source_length, where padding elements are indicated by 1s.
        """
        target_length, batch_size, embed_dim = query.size()
        mask_batch_size, source_length = key_padding_mask.size()

        assert embed_dim == self.embed_dim
        assert (
            batch_size == mask_batch_size
        ), "query and key_padding_mask batch sizes differed"

        q = self.qput_projection(query)
        q *= self.scaling

        k_input = query.permute(1, 2, 0).contiguous()  # B * C * T
        v_input = query.permute(1, 2, 0).contiguous()  # B * C * T

        k_input, v_input = self.get_compressed_projection(
            k_input, v_input, target_length
        )

        k = self.kput_projection(k_input)
        v = self.vput_projection(v_input)

        batch_heads = batch_size * self.num_heads

        q = q.contiguous().view(-1, batch_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, batch_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, batch_heads, self.head_dim).transpose(0, 1)

        source_length = k.size(1)  # T_k

        attn_weights = torch.bmm(q, k.transpose(1, 2))

        assert list(attn_weights.shape) == [batch_heads, target_length, source_length]

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(
            attn_weights
        )
        attn_weights = self.dropout(attn_weights)

        attn = torch.bmm(attn_weights, v)
        assert list(attn.shape) == [batch_heads, target_length, self.head_dim]
        attn = (
            attn.transpose(0, 1)
            .contiguous()
            .view(target_length, batch_size, self.head_dim * self.num_heads)
        )
        attn = self.output_projection(attn)

        return attn


class QuantizedMultiheadLinearAttention(MultiheadLinearAttention):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        scaling: float = 0.125,
        dropout: float = 0.1,
        compress_layer=None,
        bias: bool = True,
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            scaling=scaling,
            dropout=dropout,
            compress_layer=compress_layer,
            bias=bias,
        )
        log_class_usage(__class__)

    def get_compressed_projection(
        self, k_input: torch.Tensor, v_input: torch.Tensor, target_length: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.compress_k.in_features >= target_length
        pad = (0, self.compress_k.in_features - target_length)
        k_input = F.pad(k_input, pad)
        v_input = F.pad(v_input, pad)
        k_input = self.compress_k(k_input).permute(2, 0, 1).contiguous()
        v_input = self.compress_k(v_input).permute(2, 0, 1).contiguous()
        return k_input, v_input

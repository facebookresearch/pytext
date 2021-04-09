#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
from typing import List, Optional

import numpy as np
import torch
from pytext.utils.usage import log_class_usage
from torch import nn
from torch.nn import functional as F


class MultiheadSelfAttention(nn.Module):
    """
    This is a TorchScriptable implementation of MultiheadAttention from fairseq
    for the purposes of creating a productionized RoBERTa model. It distills just
    the elements which are required to implement the RoBERTa use cases of
    MultiheadAttention, and within that is restructured and rewritten to be able
    to be compiled by TorchScript for production use cases.

    The default constructor values match those required to import the public
    RoBERTa weights. Unless you are pretraining your own model, there's no need to
    change them.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        scaling: Optional[float] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        expected_scaling = float(1 / math.sqrt(self.head_dim))

        # for backward compatibility with previous default
        if not scaling and self.head_dim == 64:
            scaling = 0.125

        if not scaling:
            raise Exception(
                f"""
                Scaling not set. Please manually set scaling for transformers with
                head_dim != 64. The suggested value in this case is {expected_scaling},
                or float(1 / math.sqrt(head_dim))
                where head_dim = embed_dim // num_heads = {self.head_dim}
                and embed_dim = {embed_dim} and num_heads = {num_heads}.
                """
            )

        self.scaling = scaling
        self.dropout = nn.Dropout(dropout)
        self.input_projection = nn.Linear(embed_dim, 3 * embed_dim)
        self.output_projection = nn.Linear(embed_dim, embed_dim)
        log_class_usage(__class__)

    def prune_multi_heads(self, heads: List[int]):
        mask = torch.ones(self.num_heads * 3, self.head_dim)
        for head in heads:
            mask[head] = 0
            mask[head + self.num_heads] = 0
            mask[head + self.num_heads * 2] = 0
        mask = mask.view(-1).contiguous().eq(1)
        if torch.onnx.is_in_onnx_export():
            index = np.arange(len(mask), dtype=np.int64)
            index = torch.from_numpy(index).to(device=mask.device)
        else:
            index = torch.arange(len(mask), dtype=torch.long, device=mask.device)

        # Prune linear layers
        self.input_projection = self._prune_linear_layer(
            self.input_projection, index[mask], dim=0
        )
        self.output_projection = self._prune_linear_layer(
            self.output_projection,
            index[0 : self.num_heads * self.head_dim][
                mask[0 : self.num_heads * self.head_dim]
            ],
            dim=1,
        )
        # Update hyper params
        self.num_heads = self.num_heads - len(heads)

    def _prune_linear_layer(
        self, layer: torch.nn.Linear, index: torch.Tensor, dim: int = 0
    ):
        """
        Prune a linear layer (a model parameters) to keep only entries in index.
        Return the pruned layer as a new layer with requires_grad=True.
        Used to remove heads.
        """
        index = index.to(layer.weight.device)
        W = layer.weight.index_select(dim, index).clone().detach()
        if layer.bias is not None:
            if dim == 1:
                b = layer.bias.clone().detach()
            else:
                b = layer.bias[index].clone().detach()
        new_size = list(layer.weight.size())
        new_size[dim] = len(index)
        new_layer = torch.nn.Linear(
            new_size[1], new_size[0], bias=layer.bias is not None
        ).to(layer.weight.device)
        new_layer.weight.requires_grad = False
        new_layer.weight.copy_(W.contiguous())
        new_layer.weight.requires_grad = True
        if layer.bias is not None:
            new_layer.bias.requires_grad = False
            new_layer.bias.copy_(b.contiguous())
            new_layer.bias.requires_grad = True
        return new_layer

    def forward(self, query, key_padding_mask):
        """Input shape: Time x Batch x Channel
        Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x source_length, where padding elements are indicated by 1s.
        """
        target_length, batch_size, embed_dim = query.size()
        mask_batch_size, source_length = key_padding_mask.size()

        torch._assert(embed_dim == self.embed_dim, "query embed dim doesn't match")
        torch._assert(
            batch_size == mask_batch_size,
            "query and key_padding_mask batch sizes differed",
        )

        # input projection
        projection = self.input_projection(query)
        q, k, v = projection.chunk(3, dim=-1)
        q = self.scaling * q

        batch_heads = batch_size * self.num_heads

        q = q.contiguous().view(-1, batch_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, batch_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, batch_heads, self.head_dim).transpose(0, 1)

        torch._assert(
            k.size(1) == source_length, "key size should be equal to source length"
        )

        attn_weights = torch.bmm(q, k.transpose(1, 2))

        torch._assert(attn_weights.dim() == 3, "Unexpected attn_weights dim")
        torch._assert(
            attn_weights.size(0) == batch_heads,
            "attn_weights shape didn't match for batch heads",
        )
        torch._assert(
            attn_weights.size(1) == target_length,
            "attn_weights shape didn't match for target length",
        )
        torch._assert(
            attn_weights.size(2) == source_length,
            "attn_weights shape didn't match for source length",
        )

        # don't attend to padding symbols
        attn_weights = attn_weights.view(
            batch_size, self.num_heads, target_length, source_length
        )
        attn_weights = attn_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
        )
        attn_weights = attn_weights.view(batch_heads, target_length, source_length)

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(
            attn_weights
        )
        attn_weights = self.dropout(attn_weights)

        attn = torch.bmm(attn_weights, v)

        torch._assert(
            attn.dim() == 3,
            "unexpected attn dim size",
        )
        torch._assert(
            attn.size(0) == batch_heads,
            "attn shape didn't match for batch heads",
        )
        torch._assert(
            attn.size(1) == target_length,
            "attn shape didn't match for target length",
        )
        torch._assert(
            attn.size(2) == self.head_dim,
            "attn shape didn't match for head dim",
        )
        attn = (
            attn.transpose(0, 1)
            .contiguous()
            .view(target_length, batch_size, self.head_dim * self.num_heads)
        )
        attn = self.output_projection(attn)

        return attn

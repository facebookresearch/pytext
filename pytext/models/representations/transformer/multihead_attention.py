#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
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
        scaling: float = 0.125,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = scaling
        self.dropout = nn.Dropout(dropout)
        self.input_projection = nn.Linear(embed_dim, 3 * embed_dim)
        self.output_projection = nn.Linear(embed_dim, embed_dim)

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

        # input projection
        projection = self.input_projection(query)
        q, k, v = projection.chunk(3, dim=-1)
        q *= self.scaling

        batch_heads = batch_size * self.num_heads

        q = q.contiguous().view(-1, batch_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, batch_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, batch_heads, self.head_dim).transpose(0, 1)

        assert k.size(1) == source_length

        attn_weights = torch.bmm(q, k.transpose(1, 2))

        assert list(attn_weights.shape) == [batch_heads, target_length, source_length]

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
        assert list(attn.shape) == [batch_heads, target_length, self.head_dim]
        attn = (
            attn.transpose(0, 1).contiguous().view(target_length, batch_size, embed_dim)
        )
        attn = self.output_projection(attn)

        return attn

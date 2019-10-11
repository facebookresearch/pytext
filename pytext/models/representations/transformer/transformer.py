#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List, Optional

import torch
from torch import nn

from .multihead_attention import MultiheadSelfAttention
from .positional_embedding import PositionalEmbedding
from .residual_mlp import ResidualMLP


DEFAULT_EMBEDDING_DIM = 768
DEFAULT_VOCAB_SIZE = 50265
DEFAULT_PADDING_IDX = 1
DEFAULT_NUM_LAYERS = 12
DEFAULT_MAX_SEQUENCE_LENGTH = 514
DEFAULT_NUM_ATTENTION_HEADS = 12


class TransformerLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        attention: Optional[MultiheadSelfAttention] = None,
        residual_mlp: Optional[ResidualMLP] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.attention = attention or MultiheadSelfAttention(
            embedding_dim, num_heads=DEFAULT_NUM_ATTENTION_HEADS
        )
        self.residual_mlp = residual_mlp or ResidualMLP(
            embedding_dim, hidden_dims=[embedding_dim * 4]
        )

        self.attention_layer_norm = nn.LayerNorm(embedding_dim)
        self.final_layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, input, key_padding_mask):
        attention = self.attention(input, key_padding_mask)
        attention = self.dropout(attention)
        biased_input = input + attention
        biased_input = self.attention_layer_norm(biased_input)

        biased = self.residual_mlp(biased_input)
        return self.final_layer_norm(biased)


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        padding_idx: int = DEFAULT_PADDING_IDX,
        max_seq_len: int = DEFAULT_MAX_SEQUENCE_LENGTH,
        layers: List[TransformerLayer] = (),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.padding_idx = padding_idx
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx)
        self.layers = nn.ModuleList(
            layers
            or [TransformerLayer(embedding_dim) for _ in range(DEFAULT_NUM_LAYERS)]
        )
        self.positional_embedding = PositionalEmbedding(
            max_seq_len, embedding_dim, padding_idx
        )
        self.embedding_layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> List[torch.Tensor]:
        # compute padding mask. This is needed for multi-head attention
        padding_mask = tokens.eq(self.padding_idx)

        embedded = self.token_embedding(tokens)
        embedded_positions = self.positional_embedding(tokens)

        normed = self.embedding_layer_norm(embedded + embedded_positions)
        normed = self.dropout(normed)
        # account for padding while computing the representation
        padded_normed = normed * (1 - padding_mask.unsqueeze(-1).type_as(normed))

        # B x T x C -> T x B x C
        encoded = padded_normed.transpose(0, 1)

        states = [encoded]

        for layer in self.layers:
            encoded = layer(encoded, padding_mask)
            states.append(encoded)

        # states are returned as T x B x C
        # commonly you can retrieve a single "sentence representation" as
        # states[-1].transpose(0, 1)
        return states

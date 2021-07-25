#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List, Optional

import torch
import torch.nn.functional as F
from pytext.utils.usage import log_class_usage
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

# The logic in this Pytorch module is mirrored in custom operators such as
# FasterTransformer and DeepSpeed
# If you change the logic here, please ensure appropriate alignment of the
# corresponding custom operators.


class TransformerLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        attention: Optional[MultiheadSelfAttention] = None,
        residual_mlp: Optional[ResidualMLP] = None,
        dropout: float = 0.1,
        normalize_before: bool = False,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.attention = attention or MultiheadSelfAttention(
            embedding_dim, num_heads=DEFAULT_NUM_ATTENTION_HEADS
        )
        self.residual_mlp = residual_mlp or ResidualMLP(
            embedding_dim,
            hidden_dims=[embedding_dim * 4],
            add_residual=not normalize_before,
        )

        self.attention_layer_norm = nn.LayerNorm(embedding_dim)
        self.final_layer_norm = nn.LayerNorm(embedding_dim)
        self.normalize_before = normalize_before
        log_class_usage(__class__)

    def forward(self, input, key_padding_mask):
        # Using hasattr to make it backward compatible with models
        # which were trained before attribute was added.
        if not hasattr(self, "normalize_before"):
            self.normalize_before = False
        if self.normalize_before:
            x = self.attention_layer_norm(input)
            attention = self.attention(x, key_padding_mask)
            attention = self.dropout(attention)
            biased_input = input + attention
            x = self.final_layer_norm(biased_input)
            return self.residual_mlp(x) + biased_input
        else:
            attention = self.attention(input, key_padding_mask)
            attention = self.dropout(attention)
            biased_input = input + attention
            biased_input = self.attention_layer_norm(biased_input)

            biased = self.residual_mlp(biased_input)
            return self.final_layer_norm(biased)


class Transformer(nn.Module):
    def __init__(
        self,
        token_embedding,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        padding_idx: int = DEFAULT_PADDING_IDX,
        max_seq_len: int = DEFAULT_MAX_SEQUENCE_LENGTH,
        layers: List[TransformerLayer] = (),
        dropout: float = 0.1,
        normalize_before: bool = False,
    ):
        super().__init__()
        self.padding_idx = padding_idx
        self.token_embedding = token_embedding
        self.layers = nn.ModuleList(
            layers
            or [TransformerLayer(embedding_dim) for _ in range(DEFAULT_NUM_LAYERS)]
        )
        self.positional_embedding = PositionalEmbedding(
            max_seq_len, embedding_dim, padding_idx
        )
        self.embedding_layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.normalize_before = normalize_before
        log_class_usage(__class__)

    def forward(self, tokens: torch.Tensor) -> List[torch.Tensor]:
        # compute padding mask. This is needed for multi-head attention
        padding_mask = tokens.eq(self.padding_idx)

        token_embeddings = self.token_embedding(tokens)
        embedded_positions = self.positional_embedding(tokens)

        embedded = token_embeddings + embedded_positions

        # Using hasattr to make it backward compatible with models
        # which were trained before attribute was added.
        if not hasattr(self, "normalize_before"):
            self.normalize_before = False
        if not self.normalize_before:
            embedded = self.embedding_layer_norm(embedded)
        embedded = self.dropout(embedded)
        # account for padding while computing the representation
        padded_embedded = embedded * (1 - padding_mask.unsqueeze(-1).type_as(embedded))

        # B x T x C -> T x B x C
        encoded = padded_embedded.transpose(0, 1)

        states = [encoded]

        for layer in self.layers:
            encoded = layer(encoded, padding_mask)
            states.append(encoded)

        if self.normalize_before:
            for i, state in enumerate(states):
                states[i] = self.embedding_layer_norm(state)

        # states are returned as T x B x C
        # commonly you can retrieve a single "sentence representation" as
        # states[-1].transpose(0, 1)
        return states


class SELFIETransformer(Transformer):
    def forward(
        self, tokens: torch.Tensor, dense: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        # compute padding mask. This is needed for multi-head attention
        padding_mask = tokens.eq(self.padding_idx)

        embedded = self.token_embedding(tokens)
        embedded_positions = self.positional_embedding(tokens)

        normed = self.embedding_layer_norm(embedded + embedded_positions)
        normed = self.dropout(normed)
        # account for padding while computing the representation
        padded_normed = normed * (1 - padding_mask.unsqueeze(-1).type_as(normed))

        # Selfie transformer prepends dense input as first token.
        # Dim of dense must be <= embedding_dim, for now
        for i in range(len(dense)):
            padded_dense = F.pad(
                dense[i], (0, embedded.size(2) - dense[i].size(1), 0, 0), value=1.0
            )
            padded_normed = torch.cat([padded_dense.unsqueeze(1), padded_normed], dim=1)
            padding_mask = F.pad(padding_mask, (1, 0, 0, 0), value=0.0)

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

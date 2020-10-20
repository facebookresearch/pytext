#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch.nn as nn
from pytext.config import ConfigBase
from pytext.models.module import Module
from pytext.utils.usage import log_class_usage
from torch import Tensor

from .multihead_attention import MultiheadSelfAttention
from .residual_mlp import ResidualMLP
from .transformer import TransformerLayer


class TransformerRepresentation(Module):
    """
    Representation consisting of stacked multi-head self-attention
    and position-wise feed-forward layers. Unlike `Transformer`, we assume
    inputs are already embedded, thus this representation can be used as
    a drop-in replacement for other temporal representations over
    text inputs (e.g., `BiLSTM` and `DeepCNNDeepCNNRepresentation`).
    """

    class Config(ConfigBase):
        num_layers: int = 3
        num_attention_heads: int = 4
        ffnn_embed_dim: int = 32
        dropout: float = 0.0

    def __init__(self, config: Config, embed_dim: int) -> None:
        super().__init__()

        self.layers = nn.ModuleList(
            [
                self._create_transformer_layer(config, embed_dim)
                for _ in range(config.num_layers)
            ]
        )

        log_class_usage(__class__)

    def _create_transformer_layer(self, config: Config, embed_dim: int):
        return TransformerLayer(
            embedding_dim=embed_dim,
            attention=MultiheadSelfAttention(
                embed_dim=embed_dim, num_heads=config.num_attention_heads
            ),
            residual_mlp=ResidualMLP(
                input_dim=embed_dim,
                hidden_dims=[config.ffnn_embed_dim],
                dropout=config.dropout,
            ),
        )

    def forward(self, embedded_tokens: Tensor, padding_mask: Tensor) -> Tensor:
        """
        Forward inputs through the transformer layers.

        Args:
            embedded_tokens (B x T x H): Tokens previously encoded with token,
            positional, and segment embeddings.
            padding_mask (B x T): Boolean mask specifying token positions that
            self-attention should not operate on.

        Returns:
            last_state (B x T x H): Final transformer layer state.
        """

        last_state = embedded_tokens.transpose(0, 1)
        for layer in self.layers:
            last_state = layer(last_state, padding_mask)
        return last_state.transpose(0, 1)

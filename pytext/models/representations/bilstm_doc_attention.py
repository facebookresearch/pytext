#!/usr/bin/env python3
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
from pytext.config import ConfigBase
from pytext.models.decoders.mlp_decoder import MLPDecoder
from pytext.models.module import create_module
from pytext.models.representations.bilstm import BiLSTM

from .pooling import MaxPool, MeanPool, NoPool, SelfAttention
from .representation_base import RepresentationBase


class BiLSTMDocAttention(RepresentationBase):
    """Bidirectional LSTM based representation with pooling
       (e.g. max pooling or self attention)."""

    class Config(RepresentationBase.Config):
        dropout: float = 0.4
        lstm: BiLSTM.Config = BiLSTM.Config()
        pooling: Union[
            SelfAttention.Config, MaxPool.Config, MeanPool.Config, NoPool.Config
        ] = SelfAttention.Config()
        mlp_decoder: Optional[MLPDecoder.Config] = None

    def __init__(self, config: Config, embed_dim: int) -> None:
        super().__init__(config)

        self.dropout = nn.Dropout(config.dropout)

        # BiLSTM representation.
        padding_value = (
            float("-inf") if isinstance(config.pooling, MaxPool.Config) else 0.0
        )
        self.lstm = create_module(
            config.lstm, embed_dim=embed_dim, padding_value=padding_value
        )

        # Document attention.
        self.attention = (
            create_module(config.pooling, n_input=self.lstm.representation_dim)
            if config.pooling is not None
            else None
        )

        # Non-linear projection over attended representation.
        self.dense = None
        self.representation_dim = self.lstm.representation_dim
        if config.mlp_decoder:
            self.dense = MLPDecoder(
                config.mlp_decoder, in_dim=self.lstm.representation_dim
            )
            self.representation_dim = self.dense.out_dim

    def forward(
        self,
        embedded_tokens: torch.Tensor,
        seq_lengths: torch.Tensor,
        *args,
        states: torch.Tensor = None,
    ) -> Tuple[Any, Any]:
        embedded_tokens = self.dropout(embedded_tokens)

        # LSTM representation
        rep, new_state = self.lstm(embedded_tokens, seq_lengths, states)

        # Attention
        if self.attention:
            rep = self.attention(rep, seq_lengths)

        # Non-linear projection
        if self.dense:
            rep = self.dense(rep)

        return rep, new_state

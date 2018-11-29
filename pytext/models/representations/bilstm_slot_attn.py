#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Optional, Tuple

import torch
import torch.nn as nn
from pytext.config import ConfigBase
from pytext.models.decoders.mlp_decoder import MLPDecoder
from pytext.models.module import create_module
from pytext.models.representations.bilstm import BiLSTM

from .representation_base import RepresentationBase
from .slot_attention import SlotAttention


class BiLSTMSlotAttention(RepresentationBase):
    """Bidirectional LSTM based representation with slot attention."""

    class Config(RepresentationBase.Config):
        dropout: float = 0.4
        lstm: BiLSTM.Config = BiLSTM.Config()
        slot_attention: SlotAttention.Config = SlotAttention.Config()
        mlp_decoder: Optional[MLPDecoder.Config] = None

    def __init__(self, config: Config, embed_dim: int) -> None:
        super().__init__(config)

        self.dropout = nn.Dropout(config.dropout)
        self.relu = nn.ReLU()

        # BiLSTM representation.
        self.lstm = create_module(config.lstm, embed_dim=embed_dim)

        # Slot attention.
        self.attention = None
        word_representation_dim = self.lstm.representation_dim
        if config.slot_attention:
            self.attention = SlotAttention(
                config.slot_attention, self.lstm.representation_dim, batch_first=True
            )
            word_representation_dim += self.lstm.representation_dim

        # Projection over attended representation.
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
    ) -> torch.Tensor:
        rep = self.dropout(embedded_tokens)

        # LSTM representation
        rep, _ = self.lstm(embedded_tokens, seq_lengths, states)

        # Attention
        if self.attention:
            rep = self.attention(rep)

        # Non-linear projection
        return self.dense(rep) if self.dense else rep

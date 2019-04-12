#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Optional

import torch
import torch.nn as nn
from pytext.models.decoders.mlp_decoder import MLPDecoder
from pytext.models.module import create_module
from pytext.models.representations.bilstm import BiLSTM

from .representation_base import RepresentationBase
from .slot_attention import SlotAttention


class BiLSTMSlotAttention(RepresentationBase):
    """
    `BiLSTMSlotAttention` implements a multi-layer bidirectional LSTM based
    representation with attention over slots.

    Args:
        config (Config): Configuration object of type BiLSTMSlotAttention.Config.
        embed_dim (int): The number of expected features in the input.

    Attributes:
        dropout (nn.Dropout): Dropout layer preceding the LSTM.
        lstm (nn.Module): Module that implements the LSTM.
        attention (nn.Module): Module that implements the attention.
        dense (nn.Module): Module that implements the non-linear projection over
            attended representation.
        representation_dim (int): The calculated dimension of the output features
            of the `SlotAttention` representation.
    """

    class Config(RepresentationBase.Config):
        dropout: float = 0.4
        lstm: BiLSTM.Config = BiLSTM.Config()
        slot_attention: SlotAttention.Config = SlotAttention.Config()
        mlp_decoder: Optional[MLPDecoder.Config] = None

    def __init__(self, config: Config, embed_dim: int) -> None:
        super().__init__(config)

        self.dropout = nn.Dropout(config.dropout)

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
        self.representation_dim: int = self.lstm.representation_dim
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
        **kwargs,
    ) -> torch.Tensor:
        """
        Given an input batch of sequential data such as word embeddings, produces
        a bidirectional LSTM representation with or without Slot attention.

        Args:
            embedded_tokens (torch.Tensor): Input tensor of shape
                (bsize x seq_len x input_dim).
            seq_lengths (torch.Tensor): List of sequences lengths of each batch
                element.
            states (Tuple[torch.Tensor, torch.Tensor]): Tuple of tensors containing
                the initial hidden state and the cell state of each element in
                the batch. Each of these tensors have a dimension of
                (bsize x num_layers * num_directions x nhid). Defaults to `None`.

        Returns:
            torch.Tensor: Bidirectional LSTM representation of input with or
                without slot attention.

        """
        rep = self.dropout(embedded_tokens)

        # LSTM representation
        rep, state = self.lstm(rep, seq_lengths, states)

        # Attention
        if self.attention:
            rep = self.attention(rep)
        # Non-linear projection
        return (self.dense(rep) if self.dense else rep, state)

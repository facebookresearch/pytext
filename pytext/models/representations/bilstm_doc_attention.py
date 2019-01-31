#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from pytext.models.decoders.mlp_decoder import MLPDecoder
from pytext.models.module import create_module
from pytext.models.representations.bilstm import BiLSTM

from .pooling import LastTimestepPool, MaxPool, MeanPool, NoPool, SelfAttention
from .representation_base import RepresentationBase


class BiLSTMDocAttention(RepresentationBase):
    """
    `BiLSTMDocAttention` implements a multi-layer bidirectional LSTM based
    representation for documents with or without pooling. The pooling can be
    max pooling, mean pooling or self attention.

    Args:
        config (Config): Configuration object of type BiLSTMDocAttention.Config.
        embed_dim (int): The number of expected features in the input.

    Attributes:
        dropout (nn.Dropout): Dropout layer preceding the LSTM.
        lstm (nn.Module): Module that implements the LSTM.
        attention (nn.Module): Module that implements the attention or pooling.
        dense (nn.Module): Module that implements the non-linear projection over
            attended representation.
        representation_dim (int): The calculated dimension of the output features
            of the `BiLSTMDocAttention` representation.
    """

    class Config(RepresentationBase.Config):
        """
        Configuration class for `BiLSTM`.

        Attributes:
            dropout (float): Dropout probability to use. Defaults to 0.4.
            lstm (BiLSTM.Config): Config for the BiLSTM.
            pooling (ConfigBase): Config for the underlying pooling module.
            mlp_decoder (MLPDecoder.Config): Config for the non-linear
                projection module.
        """

        dropout: float = 0.4
        lstm: BiLSTM.Config = BiLSTM.Config()
        pooling: Union[
            SelfAttention.Config,
            MaxPool.Config,
            MeanPool.Config,
            NoPool.Config,
            LastTimestepPool.Config,
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
        states: Tuple[torch.Tensor, torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Given an input batch of sequential data such as word embeddings, produces
        a bidirectional LSTM representation with or without pooling of the
        sequential input and new state tensors.

        Args:
            embedded_tokens (torch.Tensor): Input tensor of shape
                (bsize x seq_len x input_dim).
            seq_lengths (torch.Tensor): List of sequences lengths of each batch element.
            states (Tuple[torch.Tensor, torch.Tensor]): Tuple of tensors containing
                the initial hidden state and the cell state of each element in
                the batch. Each of these tensors have a dimension of
                (bsize x num_layers * num_directions x nhid). Defaults to `None`.

        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: Bidirectional
                LSTM representation of input and the state of the LSTM at `t = seq_len`.
        """
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

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from pytext.config import ConfigBase
from pytext.config.module_config import SlotAttentionType
from pytext.models.module import create_module

from .bilstm import BiLSTM
from .pooling import MaxPool, MeanPool, SelfAttention
from .representation_base import RepresentationBase
from .slot_attention import SlotAttention


class BiLSTMDocSlotAttention(RepresentationBase):
    """
    `BiLSTMDocSlotAttention` implements a multi-layer bidirectional LSTM based
    representation with support for various attention mechanisms.

    In default mode, when attention configuration is not provided, it behaves
    like a multi-layer LSTM encoder and returns the output features from the
    last layer of the LSTM, for each t. When document_attention configuration is
    provided, it produces a fixed-sized document representation. When
    slot_attention configuration is provide, it attends on output of each cell
    of LSTM module to produce a fixed sized word representation.

    Args:
        config (Config): Configuration object of type
            BiLSTMDocSlotAttention.Config.
        embed_dim (int): The number of expected features in the input.

    Attributes:
        dropout (nn.Dropout): Dropout layer preceding the LSTM.
        relu (nn.ReLU): An instance of the ReLU layer.
        lstm (nn.Module): Module that implements the LSTM.
        use_doc_attention (bool): If `True`, indicates using document attention.
        doc_attention (nn.Module): Module that implements document attention.
        self.projection_d (nn.Sequential): A sequence of dense layers for
            projection over document representation.
        use_word_attention (bool): If `True`, indicates using word attention.
        word_attention (nn.Module): Module that implements word attention.
        self.projection_w (nn.Sequential): A sequence of dense layers for
            projection over word representation.
        representation_dim (int): The calculated dimension of the output features
            of the `BiLSTMDocAttention` representation.
    """

    class Config(RepresentationBase.Config, ConfigBase):
        dropout: float = 0.4
        lstm: BiLSTM.Config = BiLSTM.Config()
        pooling: Optional[
            Union[SelfAttention.Config, MaxPool.Config, MeanPool.Config]
        ] = None
        slot_attention: Optional[SlotAttention.Config] = None
        doc_mlp_layers: int = 0
        word_mlp_layers: int = 0

    def __init__(self, config: Config, embed_dim: int) -> None:
        super().__init__(config)
        self.dropout = nn.Dropout(config.dropout)
        self.relu = nn.ReLU()

        # Shared representation.
        padding_value = (
            float("-inf") if isinstance(config.pooling, MaxPool.Config) else 0.0
        )
        self.lstm = create_module(
            config.lstm, embed_dim=embed_dim, padding_value=padding_value
        )

        lstm_out_dim = self.lstm.representation_dim

        # Document projection and attention.
        self.use_doc_attention: bool = config.pooling is not None
        if config.pooling:
            self.doc_attention = (
                create_module(config.pooling, n_input=lstm_out_dim)
                if config.pooling
                else lambda x: x
            )
            layers = []
            for _ in range(config.doc_mlp_layers - 1):
                layers.extend(
                    [nn.Linear(lstm_out_dim, lstm_out_dim), self.relu, self.dropout]
                )
            layers.append(nn.Linear(lstm_out_dim, lstm_out_dim))
            self.projection_d = nn.Sequential(*layers)

        # Word projection and attention.
        self.use_word_attention = config.slot_attention is not None
        if config.slot_attention:
            word_out_dim = lstm_out_dim
            self.word_attention = lambda x: x
            if config.slot_attention.attention_type != SlotAttentionType.NO_ATTENTION:
                self.word_attention = SlotAttention(
                    config.slot_attention, lstm_out_dim, batch_first=True
                )
                word_out_dim += lstm_out_dim

            layers = [nn.Linear(word_out_dim, lstm_out_dim), self.relu, self.dropout]
            for _ in range(config.word_mlp_layers - 2):
                layers.extend(
                    [nn.Linear(lstm_out_dim, lstm_out_dim), self.relu, self.dropout]
                )
            layers.append(nn.Linear(lstm_out_dim, lstm_out_dim))
            self.projection_w = nn.Sequential(*layers)

        # Set the representation dimension attribute.
        self.representation_dim = (
            self.doc_representation_dim
        ) = self.word_representation_dim = lstm_out_dim

    def forward(
        self,
        embedded_tokens: torch.Tensor,
        seq_lengths: torch.Tensor,
        *args,
        states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Given an input batch of sequential data such as word embeddings, produces
        a bidirectional LSTM representation the appropriate attention.

        Args:
            embedded_tokens (torch.Tensor): Input tensor of shape
                (bsize x seq_len x input_dim).
            seq_lengths (torch.Tensor): List of sequences lengths of each batch
                element.
            states (Tuple[torch.Tensor, torch.Tensor]): Tuple of tensors
                containing the initial hidden state and the cell state of each
                element in the batch. Each of these tensors have a dimension of
                (bsize x num_layers * num_directions x nhid). Defaults to `None`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                Tensors containing the document and the word representation of
                the input.
        """
        # Shared layers
        lstm_output, new_state = self.lstm(embedded_tokens, seq_lengths, states)

        # Default doc representation is hidden state of last cell of LSTM.
        # Default word representation is the output state of each cell of LSTM.
        outputs = [
            new_state[0].contiguous().view(-1, self.doc_representation_dim),
            lstm_output,
        ]
        if self.use_doc_attention:
            outputs[0] = self.projection_d(self.doc_attention(lstm_output))
        if self.use_word_attention:
            outputs[1] = self.projection_w(self.word_attention(lstm_output))

        return outputs[0], outputs[1], new_state

#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .self_attention import SelfAttention
from .representation_base import RepresentationBase


class BiLSTMSelfAttention(RepresentationBase):
    """Bidirectional LSTM based representation with self attention."""

    def __init__(
        self,
        token_embeddings_dim: int,
        lstm_hidden_dim: int,
        lstm_num_layers: int,
        dropout_ratio: float,
        self_attn_dim: int,
        projection_dim: int = None,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()

        self.projection = projection_dim is not None
        self.dropout = nn.Dropout(dropout_ratio)
        seq_in_size = lstm_hidden_dim * 2 if bidirectional is True else lstm_hidden_dim
        self.lstm = nn.LSTM(
            token_embeddings_dim,
            lstm_hidden_dim,
            num_layers=lstm_num_layers,
            bidirectional=bidirectional,
        )
        self.attention = (
            SelfAttention(seq_in_size, self_attn_dim, dropout_ratio)
            if self_attn_dim > 0
            else None
        )
        if self.projection:
            self.relu = nn.ReLU()
            self.dense = nn.Sequential(
                nn.Linear(seq_in_size, projection_dim), self.relu)
            self.representation_dim = projection_dim
        else:
            self.representation_dim = seq_in_size

    def forward(self, tokens: torch.Tensor, tokens_lens: torch.Tensor) -> torch.Tensor:
        rep = self.dropout(tokens)
        tokens_lens = tokens_lens.int()
        rnn_input = pack_padded_sequence(rep, tokens_lens, batch_first=True)
        rep, _ = self.lstm(rnn_input)
        rep, _ = pad_packed_sequence(
            rep, padding_value=0.0, batch_first=True, total_length=tokens.size(1)
        )
        if self.attention:
            rep = self.attention(rep)
        if self.projection:
            return self.dense(rep)
        else:
            return rep

#!/usr/bin/env python3

import torch
import torch.nn as nn
from pytext.config import ConfigBase
from pytext.config.module_config import LSTMParams
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .representation_base import RepresentationBase
from .self_attention import SelfAttention


class BiLSTMSelfAttention(RepresentationBase):
    """Bidirectional LSTM based representation with self attention."""

    class Config(ConfigBase):
        bidirectional: bool = True
        dropout: float = 0.4
        self_attn_dim: int = 64
        lstm: LSTMParams = LSTMParams()

    def __init__(self, config: Config, embed_dim: int) -> None:
        super().__init__(config)
        self.projection = config.lstm.projection_dim is not None
        self.dropout = nn.Dropout(config.dropout)
        seq_in_size = (
            config.lstm.lstm_dim * 2
            if config.bidirectional is True
            else config.lstm.lstm_dim
        )
        self.lstm = nn.LSTM(
            embed_dim,
            config.lstm.lstm_dim,
            num_layers=config.lstm.num_layers,
            bidirectional=config.bidirectional,
        )
        self.attention = (
            SelfAttention(seq_in_size, config.self_attn_dim, config.dropout)
            if config.self_attn_dim > 0
            else None
        )
        if self.projection:
            self.relu = nn.ReLU()
            self.dense = nn.Sequential(
                nn.Linear(seq_in_size, config.lstm.projection_dim), self.relu
            )
            self.representation_dim = config.lstm.projection_dim
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

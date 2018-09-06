#!/usr/bin/env python3

import torch
import torch.nn as nn
from pytext.config import ConfigBase
from pytext.config.module_config import LSTMParams, SlotAttentionType
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .representation_base import RepresentationBase
from .slot_attention import SlotAttention


class BiLSTMSlotAttention(RepresentationBase):
    """Bidirectional LSTM based representation with slot attention."""

    class Config(ConfigBase):
        bidirectional: bool = True
        dropout: float = 0.4
        lstm: LSTMParams = LSTMParams()
        slot_attn_dim: int = 64
        slot_attention_type: SlotAttentionType = SlotAttentionType.NO_ATTENTION

    def __init__(self, config: Config, embed_dim: int) -> None:
        super().__init__(config)

        self.dropout = nn.Dropout(config.dropout)
        self.relu = nn.ReLU()

        self.lstm = nn.LSTM(
            embed_dim,
            config.lstm.lstm_dim,
            num_layers=config.lstm.num_layers,
            bidirectional=config.bidirectional,
        )

        self.attention = None
        seq_in_size = (
            config.lstm.lstm_dim * 2 if config.bidirectional else config.lstm.lstm_dim
        )
        word_input = seq_in_size
        if (
            config.slot_attn_dim > 0
            and config.slot_attention_type != SlotAttentionType.NO_ATTENTION
        ):
            self.attention = SlotAttention(
                config.slot_attention_type,
                config.slot_attn_dim,
                seq_in_size,
                batch_first=True,
            )
            word_input += seq_in_size

        self.dense = nn.Sequential(
            self.dropout,
            nn.Linear(word_input, seq_in_size),
            self.relu,
            self.dropout,
            nn.Linear(seq_in_size, seq_in_size),
            self.relu,
        )

        self.representation_dim = seq_in_size

    def forward(
        self, embedded_tokens: torch.Tensor, seq_lengths: torch.Tensor, *args
    ) -> torch.Tensor:
        rep = self.dropout(embedded_tokens)
        rnn_input = pack_padded_sequence(rep, seq_lengths.int(), batch_first=True)
        rep, _ = self.lstm(rnn_input)
        rep, _ = pad_packed_sequence(
            rep,
            padding_value=0.0,
            batch_first=True,
            total_length=embedded_tokens.size(1),
        )
        if self.attention:
            rep = self.attention(rep)
        return self.dense(rep)

#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from pytext.config.module_config import SlotAttentionType
from .slot_attention import SlotAttention
from .representation_base import RepresentationBase


class BiLSTMSlotAttention(RepresentationBase):
    """Bidirectional LSTM based representation with slot attention."""

    def __init__(
        self,
        token_embeddings_dim: int,
        lstm_hidden_dim: int,
        lstm_num_layers: int,
        dropout_ratio: float,
        slot_attention_type: SlotAttentionType,
        slot_attn_dim: int,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout_ratio)
        self.relu = nn.ReLU()

        self.lstm = nn.LSTM(
            token_embeddings_dim,
            lstm_hidden_dim,
            num_layers=lstm_num_layers,
            bidirectional=bidirectional,
        )

        self.attention = None
        seq_in_size = lstm_hidden_dim * 2 if bidirectional is True else lstm_hidden_dim
        word_input = seq_in_size
        if slot_attn_dim > 0 and slot_attention_type != SlotAttentionType.NO_ATTENTION:
            self.attention = SlotAttention(
                slot_attention_type, slot_attn_dim, seq_in_size, batch_first=True
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
        return self.dense(rep)

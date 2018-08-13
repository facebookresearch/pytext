#!/usr/bin/env python3

from typing import List

import torch
import torch.nn as nn
from pytext.config.module_config import SlotAttentionType
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .representation_base import RepresentationBase
from .self_attention import SelfAttention
from .slot_attention import SlotAttention


class JointBLSTMRepresentation(RepresentationBase):
    def __init__(
        self,
        token_embeddings_dim: int,
        lstm_hidden_dim: int,
        num_lstm_layers: int,
        lstm_bidirectional: float,
        dropout_ratio: float,
        self_attn_dim: int,
        slot_attn_dim: int,
        slot_attention_type: SlotAttentionType,
        doc_class_num: int,
    ) -> None:
        super().__init__()

        seq_in_size = 2 * lstm_hidden_dim
        self.lstm = nn.LSTM(
            token_embeddings_dim,
            lstm_hidden_dim,
            num_layers=num_lstm_layers,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout_ratio)
        self.relu = nn.ReLU()

        self.projection_d = nn.Sequential(
            nn.Linear(seq_in_size, seq_in_size),
            self.relu,
            self.dropout,
            nn.Linear(seq_in_size, seq_in_size),
            self.relu,
            self.dropout,
        )
        self.doc_attention = SelfAttention(seq_in_size, self_attn_dim, dropout_ratio)

        self.word_attention = None
        word_input = seq_in_size
        if slot_attention_type != SlotAttentionType.NO_ATTENTION:
            self.word_attention = SlotAttention(
                slot_attention_type, self_attn_dim, seq_in_size, batch_first=True
            )
            word_input += seq_in_size

        self.projection_w = nn.Sequential(
            nn.Linear(word_input, seq_in_size),
            self.relu,
            self.dropout,
            nn.Linear(seq_in_size, seq_in_size),
            self.relu,
            self.dropout,
        )

    def forward(
        self, tokens: torch.Tensor, tokens_lens: torch.Tensor
    ) -> List[torch.Tensor]:
        # tokens dim: (bsz, max_seq_len)
        # Shared layers
        max_seq_len = tokens.size()[1]
        tokens = self.dropout(tokens)
        tokens_lens = tokens_lens.int()
        lstm_input = pack_padded_sequence(tokens, tokens_lens, batch_first=True)
        lstm_out, _ = self.lstm(lstm_input)
        lstm_out, _ = pad_packed_sequence(
            lstm_out, padding_value=0.0, batch_first=True, total_length=max_seq_len
        )  # Make sure the output from LSTM is padded to input's sequence length.

        # Doc self attention + output layer
        doc_rep = self.projection_d(self.doc_attention(lstm_out))

        # Word output layer
        word_input = lstm_out
        if self.word_attention:
            word_input = self.word_attention(word_input)

        word_rep = self.projection_w(word_input)

        return [doc_rep, word_rep]

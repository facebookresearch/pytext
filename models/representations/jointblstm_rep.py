#!/usr/bin/env python3

from typing import List

import torch
import torch.nn as nn
from pytext.config import ConfigBase
from pytext.config.module_config import LSTMParams, SlotAttentionType
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .representation_base import RepresentationBase
from .self_attention import SelfAttention
from .slot_attention import SlotAttention


class JointBLSTMRepresentation(RepresentationBase):
    class Config(ConfigBase):
        dropout: float = 0.4
        attn_dim: int = 64
        lstm: LSTMParams = LSTMParams()
        slot_attention_type: SlotAttentionType = SlotAttentionType.NO_ATTENTION
        bidirectional: bool = True

    def __init__(self, config: Config, embed_dim: int) -> None:
        super().__init__(config)

        seq_in_size = config.lstm.lstm_dim * (2 if config.bidirectional else 1)
        self.lstm = nn.LSTM(
            embed_dim,
            config.lstm.lstm_dim,
            num_layers=config.lstm.num_layers,
            bidirectional=config.bidirectional,
        )

        self.dropout = nn.Dropout(config.dropout)
        self.relu = nn.ReLU()

        self.projection_d = nn.Sequential(
            nn.Linear(seq_in_size, seq_in_size),
            self.relu,
            self.dropout,
            nn.Linear(seq_in_size, seq_in_size),
            self.relu,
            self.dropout,
        )
        self.doc_attention = (
            SelfAttention(seq_in_size, config.attn_dim, config.dropout)
            if config.attn_dim > 0
            else None
        )

        self.word_attention = None
        word_input = seq_in_size
        if (
            config.attn_dim > 0
            and config.slot_attention_type != SlotAttentionType.NO_ATTENTION
        ):
            self.word_attention = SlotAttention(
                config.slot_attention_type,
                config.attn_dim,
                seq_in_size,
                batch_first=True,
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
        self.doc_representation_dim = self.word_representation_dim = seq_in_size

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
        doc_input = lstm_out
        if self.doc_attention:
            doc_input = self.doc_attention(doc_input)
        doc_rep = self.projection_d(doc_input)

        # Word output layer
        word_input = lstm_out
        if self.word_attention:
            word_input = self.word_attention(word_input)

        word_rep = self.projection_w(word_input)

        return [doc_rep, word_rep]

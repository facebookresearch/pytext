#!/usr/bin/env python3
from typing import Tuple

import torch
import torch.nn as nn
from pytext.config import ConfigBase
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .representation_base import RepresentationBase


class BiLSTM(RepresentationBase):
    """Bidirectional LSTM based document representation."""

    class Config(ConfigBase):
        dropout: float = 0.4
        lstm_dim: int = 32
        num_layers: int = 1
        bidirectional: bool = True

    def __init__(
        self, config: Config, embed_dim: int, padding_value: float = 0.0
    ) -> None:
        super().__init__(config)

        self.padding_value = padding_value
        self.dropout = nn.Dropout(config.dropout)
        self.lstm = nn.LSTM(
            embed_dim,
            config.lstm_dim,
            num_layers=config.num_layers,
            bidirectional=config.bidirectional,
        )
        self.representation_dim = config.lstm_dim * (2 if config.bidirectional else 1)

    def forward(
        self,
        embedded_tokens: torch.Tensor,
        seq_lengths: torch.Tensor,
        states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, ...]:
        embedded_tokens = self.dropout(embedded_tokens)
        if states is not None:
            # convert (h0, c0) from (bsz x seq_len x nhid) to (seq_len x bsz x nhid)
            states = (
                states[0].transpose(0, 1).contiguous(),
                states[1].transpose(0, 1).contiguous(),
            )
        rnn_input = pack_padded_sequence(
            embedded_tokens, seq_lengths.int(), batch_first=True
        )
        rep, new_state = self.lstm(rnn_input, states)
        rep, _ = pad_packed_sequence(
            rep,
            padding_value=self.padding_value,
            batch_first=True,
            total_length=embedded_tokens.size(1),
        )  # Make sure the output from LSTM is padded to input's sequence length.

        # convert states back to (bsz x seq_len x nhid) to be used in
        # data parallel model
        new_state = (new_state[0].transpose(0, 1), new_state[1].transpose(0, 1))

        return rep, new_state

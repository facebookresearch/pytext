#!/usr/bin/env python3
from typing import Tuple, Union

import torch
import torch.nn as nn
from pytext.config import ConfigBase
from pytext.config.component import create_module
from pytext.config.module_config import LSTMParams
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .pooling import MaxPool, MeanPool, NoPool, SelfAttention
from .representation_base import RepresentationBase


class BiLSTMPooling(RepresentationBase):
    """Bidirectional LSTM based representation with pooling
       (e.g. max pooling or self attention)."""

    class Config(RepresentationBase.Config, ConfigBase):
        pooling: Union[
            SelfAttention.Config, MaxPool.Config, MeanPool.Config, NoPool.Config
        ] = SelfAttention.Config()
        bidirectional: bool = True
        dropout: float = 0.4
        lstm: LSTMParams = LSTMParams()

    def __init__(self, config: Config, embed_dim: int) -> None:
        super().__init__(config)
        self.decoder = config.lstm.projection_dim is not None
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
            create_module(config.pooling, n_input=seq_in_size)
            if config.pooling is not None
            else None
        )
        if self.decoder:
            self.relu = nn.ReLU()
            self.dense = nn.Sequential(
                nn.Linear(seq_in_size, config.lstm.projection_dim), self.relu
            )
            self.representation_dim = config.lstm.projection_dim
        else:
            self.representation_dim = seq_in_size

    def forward(
        self,
        embedded_tokens: torch.Tensor,
        seq_lengths: torch.Tensor,
        dict_feat: Tuple[torch.Tensor, ...] = None,
        cap_feat: Tuple[torch.Tensor, ...] = None,
        chars: torch.Tensor = None,
        pre_trained_rep_feat: torch.Tensor = None,
        states: torch.Tensor = None,
    ) -> torch.Tensor:
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
            padding_value=0.0,
            batch_first=True,
            total_length=embedded_tokens.size(1),
        )
        # convert states back to (bsz x seq_len x nhid) to be used in
        # data parallel model
        new_state = (new_state[0].transpose(0, 1), new_state[1].transpose(0, 1))
        if self.attention:
            rep = self.attention(rep)
        if self.decoder:
            return self.dense(rep), new_state
        else:
            return rep, new_state

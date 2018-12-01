#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Optional, Tuple

import torch
import torch.nn as nn
from pytext.config import ConfigBase
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .representation_base import RepresentationBase


class BiLSTM(RepresentationBase):
    """
    `BiLSTM` implements a multi-layer bidirectional LSTM representation layer
    preceded by a dropout layer.

    Args:
        config (Config): Configuration object of type BiLSTM.Config.
        embed_dim (int): The number of expected features in the input.
        padding_value (float): Value for the padded elements. Defaults to 0.0.

    Attributes:
        padding_value (float): Value for the padded elements.
        dropout (nn.Dropout): Dropout layer preceding the LSTM.
        lstm (nn.LSTM): LSTM layer that operates on the inputs.
        representation_dim (int): The calculated dimension of the output features
            of BiLSTM.
    """

    class Config(RepresentationBase.Config, ConfigBase):
        """
        Configuration class for `BiLSTM`.

        Attributes:
            dropout (float): Dropout probability to use. Defaults to 0.4.
            lstm_dim (int): Number of features in the hidden state of the LSTM.
                Defaults to 32.
            num_layers (int): Number of recurrent layers. Eg. setting `num_layers=2`
                would mean stacking two LSTMs together to form a stacked LSTM,
                with the second LSTM taking in the outputs of the first LSTM and
                computing the final result. Defaults to 1.
            bidirectional (bool): If `True`, becomes a bidirectional LSTM. Defaults
                to `True`.
        """

        dropout: float = 0.4
        lstm_dim: int = 32
        num_layers: int = 1
        bidirectional: bool = True

    def __init__(
        self, config: Config, embed_dim: int, padding_value: float = 0.0
    ) -> None:
        super().__init__(config)

        self.padding_value: float = padding_value
        self.dropout = nn.Dropout(config.dropout)
        self.lstm = nn.LSTM(
            embed_dim,
            config.lstm_dim,
            num_layers=config.num_layers,
            bidirectional=config.bidirectional,
        )
        self.representation_dim: int = config.lstm_dim * (
            2 if config.bidirectional else 1
        )

    def forward(
        self,
        embedded_tokens: torch.Tensor,
        seq_lengths: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Given an input batch of sequential data such as word embeddings, produces
        a bidirectional LSTM representation of the sequential input and new state
        tensors.

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
                LSTM representation of input and the state of the LSTM `t = seq_len`.
                Shape of representation is (bsize x seq_len x representation_dim).
                Shape of each state is (bsize x num_layers * num_directions x nhid).

        """
        embedded_tokens = self.dropout(embedded_tokens)
        if states is not None:
            # convert (h0, c0) from (bsz x num_layers*num_directions x nhid) to
            # (num_layers*num_directions x bsz x nhid)
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

        # convert states back to (bsz x num_layers*num_directions x nhid) to be
        # used in data parallel model
        new_state = (new_state[0].transpose(0, 1), new_state[1].transpose(0, 1))

        return rep, new_state

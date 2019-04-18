#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from pytext.config import ConfigBase

from .augmented_lstm import AugmentedLSTM
from .bilstm import BiLSTM
from .representation_base import RepresentationBase


class RNNUnion(RepresentationBase):
    """
    `RNNUnion` presents a union class over all the different types of RNN's within
    PyText. This allows the user to set the rnn type as a configurable option.

    Args:
        config (Config): Configuration object of type RNNUnion.Config.
        embed_dim (int): The number of expected features in the input.
        padding_value (float): Value for the padded elements. Defaults to 0.0.

    Attributes:
        padding_value (float): Value for the padded elements.
        inner_rnn (nn.Module): PyText module representing the rnn.
        representation_dim (int): The calculated dimension of the output features
            of RNN.
    """

    class Config(RepresentationBase.Config, ConfigBase):
        """
        Configuration class for `RNNUnion`.

        Attributes:
            dropout (float): Dropout probability to use. Defaults to 0.4.
            hidden_size (int): Number of features in the hidden state of the LSTM.
                Defaults to 32.
            use_highway (bool): If `True` we append a highway network
                to the outputs of the LSTM.
            bidirectional (bool): If `True`, becomes a bidirectional LSTM. Defaults
                to `True`.
            num_layers (int): Number of recurrent layers. Eg. setting `num_layers=2`
                would mean stacking two LSTMs together to form a stacked LSTM,
                with the second LSTM taking in the outputs of the first LSTM and
                computing the final result. Defaults to 1.
            use_bias (bool): If `True` we use a bias in our LSTM calculations, otherwise
                we don't.
            rnn_type (str): The type of rnn to use. We currently support
                (auglstm | bilistm).
        """

        dropout: float = 0.0
        hidden_size: int = 32
        use_highway: bool = True
        bidirectional: bool = False
        num_layers: int = 1
        use_bias: bool = True
        rnn_type: str = "auglstm"

        def get_augmented_lstm_config(self) -> AugmentedLSTM.Config:
            augmented_lstm_config = AugmentedLSTM.Config()
            augmented_lstm_config.bidirectional = self.bidirectional
            augmented_lstm_config.dropout = self.dropout
            augmented_lstm_config.use_bias = self.use_bias
            augmented_lstm_config.num_layers = self.num_layers
            augmented_lstm_config.hidden_size = self.hidden_size
            augmented_lstm_config.use_highway = self.use_highway
            return augmented_lstm_config

        def get_bilstm_config(self) -> BiLSTM.Config:
            bilstm_config = BiLSTM.Config()
            bilstm_config.dropout = self.dropout
            bilstm_config.lstm_dim = self.hidden_size
            bilstm_config.num_layers = self.num_layers
            bilstm_config.bidirectional = self.bidirectional
            bilstm_config.bias = self.use_bias
            return bilstm_config

    def __init__(self, config: Config, input_size: int, padding_value: float = 0.0):
        super().__init__(config)
        self.config = config
        self.inner_rnn: nn.Module = None
        if config.rnn_type == "auglstm":
            self.inner_rnn = AugmentedLSTM(
                self.config.get_augmented_lstm_config(), input_size, padding_value
            )
        elif config.rnn_type == "bilstm":
            self.inner_rnn = BiLSTM(
                self.config.get_bilstm_config(), input_size, padding_value
            )
        else:
            raise ValueError(
                f"Did not understand rnn of type {config.rnn_type}."
                + "Please select from (auglstm | bilistm)"
            )
        self.representation_dim = self.inner_rnn.representation_dim

    def forward(
        self,
        embedded_tokens: torch.Tensor,
        seq_lengths: torch.Tensor,
        states: Optional[Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]]:
        """
        Given an input batch of sequential data such as word embeddings, produces
        a RNN representation of the sequential input and new state
        tensors.

        Args:
            embedded_tokens (torch.Tensor): Input tensor of shape
                (bsize x seq_len x input_dim).
            seq_lengths (torch.Tensor): List of sequences lengths of each batch element.
            states (Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]):
                Tuple of tensors containing
                the initial hidden state and possibly the cell state of each element
                in the batch. Each of these tensors have a dimension of
                (bsize x num_layers * num_directions x nhid). Defaults to `None`.

        Returns:
            Tuple[torch.Tensor, Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]]:
                RNN representation of input and the state of the RNN `t = seq_len`.
                Shape of representation is (bsize x seq_len x representation_dim).
                Shape of each state is (bsize x num_layers * num_directions x nhid).

        """
        return self.inner_rnn(embedded_tokens, seq_lengths, states)

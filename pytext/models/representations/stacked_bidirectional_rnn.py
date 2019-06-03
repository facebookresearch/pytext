#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from enum import Enum

import torch
import torch.nn as nn
from pytext.models.module import Module


class RnnType(Enum):
    RNN = "rnn"
    LSTM = "lstm"
    GRU = "gru"


RNN_TYPE_DICT = {RnnType.RNN: nn.RNN, RnnType.LSTM: nn.LSTM, RnnType.GRU: nn.GRU}


class StackedBidirectionalRNN(Module):
    """
    `StackedBidirectionalRNN` implements a multi-layer bidirectional RNN with an
    option to return outputs from all the layers of RNN.

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

    class Config(Module.Config):
        """
        Configuration class for `StackedBidirectionalRNN`.

        Attributes:
            hidden_size (int): Number of features in the hidden state of the RNN.
                Defaults to 32.
            num_layers (int): Number of recurrent layers. Eg. setting `num_layers=2`
                would mean stacking two RNNs together to form a stacked RNN,
                with the second RNN taking in the outputs of the first RNN and
                computing the final result. Defaults to 1.
            dropout (float): Dropout probability to use. Defaults to 0.4.
            bidirectional (bool): If `True`, becomes a bidirectional RNN. Defaults
                to `True`.
            rnn_type (str): Which RNN type to use. Options: "rnn", "lstm", "gru".
            concat_layers (bool): Whether to concatenate the outputs of each layer
                of stacked RNN.
        """

        hidden_size: int = 32
        num_layers: int = 1
        dropout: float = 0.0
        bidirectional: bool = True
        rnn_type: RnnType = RnnType.LSTM
        concat_layers: bool = True

    def __init__(self, config: Config, input_size: int, padding_value: float = 0.0):
        super().__init__()
        self.num_layers = config.num_layers
        self.dropout = nn.Dropout(config.dropout)
        self.concat_layers = config.concat_layers
        self.padding_value = padding_value
        self.rnns = nn.ModuleList()

        rnn_module = RNN_TYPE_DICT.get(config.rnn_type)
        assert rnn_module is not None, "rnn_cell cannot be None"
        for i in range(config.num_layers):
            input_size = input_size if i == 0 else 2 * config.hidden_size
            self.rnns.append(
                rnn_module(
                    input_size,
                    config.hidden_size,
                    num_layers=1,
                    bidirectional=config.bidirectional,
                )
            )
        self.representation_dim = (
            (config.num_layers if config.concat_layers else 1)
            * config.hidden_size
            * (2 if config.bidirectional else 1)
        )

    def forward(self, tokens, tokens_mask):
        """
        Args:
            tokens: batch, max_seq_len, hidden_size
            tokens_mask: batch, max_seq_len (1 for padding, 0 for true)
        Output:
            tokens_encoded: batch, max_seq_len, hidden_size * num_layers if
                concat_layers = True else batch, max_seq_len, hidden_size
        """
        # Sort in descending order of sequence lengths.
        seq_lengths = tokens_mask.eq(0).long().sum(1)
        seq_lengths_sorted, idx_of_sorted = torch.sort(
            seq_lengths, dim=0, descending=True
        )

        # Sort tokens and pack into padded sequence.
        tokens_sorted = tokens.index_select(0, idx_of_sorted)
        packed_tokens = nn.utils.rnn.pack_padded_sequence(
            tokens_sorted, seq_lengths_sorted, batch_first=True
        )

        # Pass tokens through stacked RNN and get all layers' output.
        outputs = [packed_tokens]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]
            rnn_input = nn.utils.rnn.PackedSequence(
                self.dropout(rnn_input.data), rnn_input.batch_sizes
            )
            outputs.append(self.rnns[i](rnn_input)[0])
        outputs = outputs[1:]  # Ignore the first element which just tokens.

        # Unpack packed RNN outouts and concatenate if necessary.
        for i in range(len(outputs)):
            outputs[i] = nn.utils.rnn.pad_packed_sequence(
                outputs[i], padding_value=self.padding_value, batch_first=True
            )[0]
        output = torch.cat(outputs, 2) if self.concat_layers else outputs[-1]

        # Restore to original order of examples.
        _, idx_of_original = torch.sort(idx_of_sorted, dim=0)
        output = output.index_select(0, idx_of_original)

        # Pad upto original batch's maximum sequence length.
        max_seq_len = tokens_mask.size(1)
        batch_size, output_seq_len, output_dim = output.size()
        if output_seq_len != max_seq_len:
            padding = torch.zeros(
                batch_size, max_seq_len - output_seq_len, output_dim
            ).type(output.data.type())
            output = torch.cat([output, padding], 1)

        return output

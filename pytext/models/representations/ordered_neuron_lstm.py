#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytext.config import ConfigBase
from pytext.models.module import Module
from pytext.utils import cuda

from .representation_base import RepresentationBase


# A single layer of an Ordered Neuron LSTM
class OrderedNeuronLSTMLayer(Module):
    def __init__(
        self, embed_dim: int, lstm_dim: int, padding_value: float, dropout: float
    ) -> None:
        super().__init__()
        self.lstm_dim = lstm_dim
        self.padding_value = padding_value
        self.dropout = nn.Dropout(dropout)

        total_size = embed_dim + lstm_dim
        self.f_gate = nn.Linear(total_size, lstm_dim)
        self.i_gate = nn.Linear(total_size, lstm_dim)
        self.o_gate = nn.Linear(total_size, lstm_dim)
        self.c_hat_gate = nn.Linear(total_size, lstm_dim)
        self.master_forget_no_cumax_gate = nn.Linear(total_size, lstm_dim)
        self.master_input_no_cumax_gate = nn.Linear(total_size, lstm_dim)

    # embedded_tokens has shape (seq length, batch size, embed size)
    # states = (hidden, context), where both hidden and context have
    #          shape (batch size, hidden size)
    def forward(
        self,
        embedded_tokens: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor],
        seq_lengths: List[int],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hidden, context = states
        batch_size = hidden.size(0)
        all_context = []
        all_hidden = []

        if self.dropout.p > 0.0:
            embedded_tokens = self.dropout(embedded_tokens)

        for batch in embedded_tokens:
            # Compute the normal LSTM gates
            combined = torch.cat((batch, hidden), 1)
            ft = self.f_gate(combined).sigmoid()
            it = self.i_gate(combined).sigmoid()
            ot = self.o_gate(combined).sigmoid()
            c_hat = self.c_hat_gate(combined).tanh()

            # Compute the master gates
            master_forget_no_cumax = self.master_forget_no_cumax_gate(combined)
            master_forget = torch.cumsum(
                F.softmax(master_forget_no_cumax, dim=1), dim=1
            )
            master_input_no_cumax = self.master_input_no_cumax_gate(combined)
            master_input = torch.cumsum(F.softmax(master_input_no_cumax, dim=1), dim=1)

            # Combine master gates with normal LSTM gates
            wt = master_forget * master_input
            f_hat_t = ft * wt + (master_forget - wt)
            i_hat_t = it * wt + (master_input - wt)

            # Compute new context and hidden using final combined gates
            context = f_hat_t * context + i_hat_t * c_hat
            hidden = ot * context
            all_context.append(context)
            all_hidden.append(hidden)

        # Compute what the final state (hidden and context for each element
        # in the batch) should be based on seq_lengths
        state_hidden = []
        state_context = []

        for i in range(batch_size):
            seq_length = seq_lengths[i]
            state_hidden.append(all_hidden[seq_length - 1][i])
            state_context.append(all_context[seq_length - 1][i])

        # Return hidden states across all time steps, and return a tuple
        # containing the hidden and context for the last time step (might
        # be different based on seq_lengths)
        return (
            torch.stack(all_hidden),
            (torch.stack(state_hidden), torch.stack(state_context)),
        )


# Ordered Neuron LSTM with any number of layers
class OrderedNeuronLSTM(RepresentationBase):
    class Config(RepresentationBase.Config, ConfigBase):
        dropout: float = 0.4
        lstm_dim: int = 32
        num_layers: int = 1

    def __init__(
        self, config: Config, embed_dim: int, padding_value: Optional[float] = 0.0
    ) -> None:
        super().__init__(config)
        self.representation_dim = config.lstm_dim
        self.padding_value = padding_value
        lstms = []
        sizes = [embed_dim] + ([config.lstm_dim] * config.num_layers)

        # Create an ONLstm for each hidden size, and chain them together
        # using lstms
        for i in range(len(sizes) - 1):
            lstm = OrderedNeuronLSTMLayer(
                sizes[i], sizes[i + 1], padding_value, config.dropout
            )
            lstms.append(lstm)

        self.lstms = nn.ModuleList(lstms)

    # rep has shape (batch size, seq length, embed dim)
    # seq_lengths has sequence lengths for each case in the batch, used to
    #   pick the last hidden and context
    # states is a tuple for initial hidden and context
    def forward(
        self,
        rep: torch.Tensor,
        seq_lengths: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if states is not None:
            # Transpose states so hidden and context both have shape
            # (num layers, batch size, lstm dim)
            states = (
                states[0].transpose(0, 1).contiguous(),
                states[1].transpose(0, 1).contiguous(),
            )
        else:
            # state has shape (num layers, batch size, lstm dim)
            state = torch.zeros(
                self.config.num_layers,
                rep.size(0),
                self.config.lstm_dim,
                device=torch.cuda.current_device() if cuda.CUDA_ENABLED else None,
            )

            states = (state, state)

        # hidden_by_layer is a list of hidden states for each layer of the
        # network, and similarly for context_by_layer
        hidden_by_layer, context_by_layer = states

        # Collect the last hidden and context for each layer
        last_hidden_by_layer = []
        last_context_by_layer = []
        rep = rep.transpose(0, 1).contiguous()

        for lstm, hidden, context in zip(self.lstms, hidden_by_layer, context_by_layer):
            state = (hidden, context)

            # We purposefully throw away new_state until we reach the top layer
            # since we only care about passing on the final hidden state
            rep, (last_hidden, last_context) = lstm(rep, state, seq_lengths)
            last_hidden_by_layer.append(last_hidden)
            last_context_by_layer.append(last_context)

        # Make rep have shape (batch size, num layers, hidden size)
        rep = rep.transpose(0, 1).contiguous()

        # Make last_hidden and last_context have shape
        # (batch size, num layers, hidden size)
        last_hidden = torch.stack(last_hidden_by_layer).transpose(0, 1)
        last_context = torch.stack(last_context_by_layer).transpose(0, 1)
        return rep, (last_hidden, last_context)

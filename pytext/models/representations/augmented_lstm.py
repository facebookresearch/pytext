#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from pytext.config import ConfigBase
from pytext.utils.usage import log_class_usage
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence, pad_packed_sequence

from .representation_base import RepresentationBase


class AugmentedLSTMCell(nn.Module):
    """
    `AugmentedLSTMCell` implements a AugmentedLSTM cell.
    Args:
        embed_dim (int): The number of expected features in the input.
        lstm_dim (int): Number of features in the hidden state of the LSTM.
        Defaults to 32.
        use_highway (bool): If `True` we append a highway network to the
        outputs of the LSTM.
        use_bias (bool): If `True` we use a bias in our LSTM calculations, otherwise
        we don't.

    Attributes:
        input_linearity (nn.Module): Fused weight matrix which
            computes a linear function over the input.
        state_linearity (nn.Module): Fused weight matrix which
            computes a linear function over the states.
    """

    def __init__(
        self, embed_dim: int, lstm_dim: int, use_highway: bool, use_bias: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.lstm_dim = lstm_dim
        self.use_highway = use_highway
        self.use_bias = use_bias

        if use_highway:
            self._highway_inp_proj_start = 5 * self.lstm_dim
            self._highway_inp_proj_end = 6 * self.lstm_dim

            # fused linearity of input to input_gate,
            # forget_gate, memory_init, output_gate, highway_gate,
            # and the actual highway value
            self.input_linearity = nn.Linear(
                self.embed_dim, self._highway_inp_proj_end, bias=self.use_bias
            )
            # fused linearity of input to input_gate,
            # forget_gate, memory_init, output_gate, highway_gate
            self.state_linearity = nn.Linear(
                self.lstm_dim, self._highway_inp_proj_start, bias=True
            )
        else:
            # If there's no highway layer then we have a standard
            # LSTM. The 4 comes from fusing input, forget, memory, output
            # gates/inputs.
            self.input_linearity = nn.Linear(
                self.embed_dim, 4 * self.lstm_dim, bias=self.use_bias
            )
            self.state_linearity = nn.Linear(
                self.lstm_dim, 4 * self.lstm_dim, bias=True
            )
        self.reset_parameters()
        log_class_usage(__class__)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.lstm_dim)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(
        self,
        x: torch.Tensor,
        states=Tuple[torch.Tensor, torch.Tensor],
        variational_dropout_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Warning: DO NOT USE THIS LAYER DIRECTLY, INSTEAD USE the AugmentedLSTM class

        Args:
            x (torch.Tensor): Input tensor of shape
                (bsize x input_dim).
            states (Tuple[torch.Tensor, torch.Tensor]): Tuple of tensors containing
                the hidden state and the cell state of each element in
                the batch. Each of these tensors have a dimension of
                (bsize x nhid). Defaults to `None`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                Returned states. Shape of each state is (bsize x nhid).

        """
        hidden_state, memory_state = states
        projected_input = self.input_linearity(x)
        projected_state = self.state_linearity(hidden_state)

        input_gate = forget_gate = memory_init = output_gate = highway_gate = None
        if self.use_highway:
            fused_op = projected_input[:, : 5 * self.lstm_dim] + projected_state
            fused_chunked = torch.chunk(fused_op, 5, 1)
            (
                input_gate,
                forget_gate,
                memory_init,
                output_gate,
                highway_gate,
            ) = fused_chunked
            highway_gate = torch.sigmoid(highway_gate)
        else:
            fused_op = projected_input + projected_state
            input_gate, forget_gate, memory_init, output_gate = torch.chunk(
                fused_op, 4, 1
            )
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        memory_init = torch.tanh(memory_init)
        output_gate = torch.sigmoid(output_gate)
        memory = input_gate * memory_init + forget_gate * memory_state
        timestep_output: torch.Tensor = output_gate * torch.tanh(memory)

        if self.use_highway:
            highway_input_projection = projected_input[
                :, self._highway_inp_proj_start : self._highway_inp_proj_end
            ]
            timestep_output = (
                highway_gate * timestep_output
                + (1 - highway_gate) * highway_input_projection  # noqa
            )
        if variational_dropout_mask is not None and self.training:
            timestep_output = timestep_output * variational_dropout_mask
        return timestep_output, memory


class AugmentedLSTMUnidirectional(nn.Module):
    """
    `AugmentedLSTMUnidirectional` implements a one-layer single directional
    AugmentedLSTM layer. AugmentedLSTM is an LSTM which optionally
    appends an optional highway network to the output layer. Furthermore the
    dropout controlls the level of variational dropout done.

    Args:
        embed_dim (int): The number of expected features in the input.
        lstm_dim (int): Number of features in the hidden state of the LSTM.
            Defaults to 32.
        go_forward (bool): Whether to compute features left to right (forward)
            or right to left (backward).
        recurrent_dropout_probability (float): Variational dropout probability
            to use. Defaults to 0.0.
        use_highway (bool): If `True` we append a highway network to the
            outputs of the LSTM.
        use_input_projection_bias (bool): If `True` we use a bias in
            our LSTM calculations, otherwise we don't.

    Attributes:
        cell (AugmentedLSTMCell): AugmentedLSTMCell that is applied at every timestep.
    """

    def __init__(
        self,
        embed_dim: int,
        lstm_dim: int,
        go_forward: bool = True,
        recurrent_dropout_probability: float = 0.0,
        use_highway: bool = True,
        use_input_projection_bias: bool = True,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.lstm_dim = lstm_dim

        self.go_forward = go_forward
        self.use_highway = use_highway
        self.recurrent_dropout_probability = recurrent_dropout_probability

        self.cell = AugmentedLSTMCell(
            self.embed_dim, self.lstm_dim, self.use_highway, use_input_projection_bias
        )
        log_class_usage(__class__)

    def get_dropout_mask(
        self, dropout_probability: float, tensor_for_masking: torch.Tensor
    ) -> torch.Tensor:
        binary_mask = (torch.rand(tensor_for_masking.size()) > dropout_probability).to(
            tensor_for_masking.device
        )
        # Scale mask by 1/keep_prob to preserve output statistics.
        dropout_mask = binary_mask.float().div(1.0 - dropout_probability)
        return dropout_mask

    def forward(
        self,
        inputs: PackedSequence,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[PackedSequence, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Warning: DO NOT USE THIS LAYER DIRECTLY, INSTEAD USE the AugmentedLSTM class

        Given an input batch of sequential data such as word embeddings, produces
        a single layer unidirectional AugmentedLSTM representation of the sequential
        input and new state tensors.

        Args:
            inputs (PackedSequence): Input tensor of shape
                (bsize x seq_len x input_dim).
            states (Tuple[torch.Tensor, torch.Tensor]): Tuple of tensors containing
                the initial hidden state and the cell state of each element in
                the batch. Each of these tensors have a dimension of
                (1 x bsize x num_directions * nhid). Defaults to `None`.

        Returns:
            Tuple[PackedSequence, Tuple[torch.Tensor, torch.Tensor]]:
                AgumentedLSTM representation of input and the
                state of the LSTM `t = seq_len`.
                Shape of representation is (bsize x seq_len x representation_dim).
                Shape of each state is (1 x bsize x nhid).

        """
        sequence_tensor, batch_lengths = pad_packed_sequence(inputs, batch_first=True)
        batch_size = sequence_tensor.size()[0]
        total_timesteps = sequence_tensor.size()[1]
        output_accumulator = sequence_tensor.new_zeros(
            batch_size, total_timesteps, self.lstm_dim
        )
        if states is None:
            full_batch_previous_memory = sequence_tensor.new_zeros(
                batch_size, self.lstm_dim
            )
            full_batch_previous_state = sequence_tensor.data.new_zeros(
                batch_size, self.lstm_dim
            )
        else:
            full_batch_previous_state = states[0].squeeze(0)
            full_batch_previous_memory = states[1].squeeze(0)
        current_length_index = batch_size - 1 if self.go_forward else 0
        if self.recurrent_dropout_probability > 0.0:
            dropout_mask = self.get_dropout_mask(
                self.recurrent_dropout_probability, full_batch_previous_memory
            )
        else:
            dropout_mask = None

        for timestep in range(total_timesteps):
            index = timestep if self.go_forward else total_timesteps - timestep - 1

            if self.go_forward:
                while batch_lengths[current_length_index] <= index:
                    current_length_index -= 1
            # If we're going backwards, we are _picking up_ more indices.
            else:
                # First conditional: Are we already at the maximum
                # number of elements in the batch?
                # Second conditional: Does the next shortest
                # sequence beyond the current batch
                # index require computation use this timestep?
                while (
                    current_length_index < (len(batch_lengths) - 1)
                    and batch_lengths[current_length_index + 1] > index
                ):
                    current_length_index += 1

            previous_memory = full_batch_previous_memory[
                0 : current_length_index + 1
            ].clone()
            previous_state = full_batch_previous_state[
                0 : current_length_index + 1
            ].clone()
            timestep_input = sequence_tensor[0 : current_length_index + 1, index]
            timestep_output, memory = self.cell(
                timestep_input,
                (previous_state, previous_memory),
                dropout_mask[0 : current_length_index + 1]
                if dropout_mask is not None
                else None,
            )
            full_batch_previous_memory = full_batch_previous_memory.data.clone()
            full_batch_previous_state = full_batch_previous_state.data.clone()
            full_batch_previous_memory[0 : current_length_index + 1] = memory
            full_batch_previous_state[0 : current_length_index + 1] = timestep_output
            output_accumulator[0 : current_length_index + 1, index, :] = timestep_output

        output_accumulator = pack_padded_sequence(
            output_accumulator, batch_lengths, batch_first=True
        )

        # Mimic the pytorch API by returning state in the following shape:
        # (num_layers * num_directions, batch_size, lstm_dim). As this
        # LSTM cannot be stacked, the first dimension here is just 1.
        final_state = (
            full_batch_previous_state.unsqueeze(0),
            full_batch_previous_memory.unsqueeze(0),
        )
        return output_accumulator, final_state


class AugmentedLSTM(RepresentationBase):
    """
    `AugmentedLSTM` implements a generic AugmentedLSTM representation layer.
    AugmentedLSTM is an LSTM which optionally appends
    an optional highway network to the output layer.
    Furthermore the dropout controlls the level of variational dropout done.

    Args:
        config (Config): Configuration object of type BiLSTM.Config.
        embed_dim (int): The number of expected features in the input.
        padding_value (float): Value for the padded elements. Defaults to 0.0.

    Attributes:
        padding_value (float): Value for the padded elements.
        forward_layers (nn.ModuleList): A module list of unidirectional AugmentedLSTM
            layers moving forward in time.
        backward_layers (nn.ModuleList): A module list of unidirectional AugmentedLSTM
            layers moving backward in time.
        representation_dim (int): The calculated dimension of the output features
            of AugmentedLSTM.
    """

    class Config(RepresentationBase.Config, ConfigBase):
        """
        Configuration class for `AugmentedLSTM`.

        Attributes:
            dropout (float): Variational dropout probability to use.
                Defaults to 0.0.
            lstm_dim (int): Number of features in the hidden state of the LSTM.
                Defaults to 32.
            num_layers (int): Number of recurrent layers. Eg. setting `num_layers=2`
                would mean stacking two LSTMs together to form a stacked LSTM,
                with the second LSTM taking in the outputs of the first LSTM and
                computing the final result. Defaults to 1.
            bidirectional (bool): If `True`, becomes a bidirectional LSTM. Defaults
                to `True`.
            use_highway (bool): If `True` we append a highway network
                to the outputs of the LSTM.
            use_bias (bool): If `True` we use a bias in our LSTM calculations, otherwise
                we don't.
        """

        dropout: float = 0.0
        lstm_dim: int = 32
        use_highway: bool = True
        bidirectional: bool = False
        num_layers: int = 1
        use_bias: bool = False

    def __init__(
        self, config: Config, embed_dim: int, padding_value: float = 0.0
    ) -> None:
        super().__init__(config)

        self.config = config
        self.embed_dim = embed_dim
        self.padding_value = padding_value
        self.lstm_dim = self.config.lstm_dim
        self.num_layers = self.config.num_layers
        self.bidirectional = self.config.bidirectional
        self.dropout = self.config.dropout
        self.use_highway = self.config.use_highway
        self.use_bias = self.config.use_bias

        num_directions = int(self.bidirectional) + 1
        self.forward_layers = nn.ModuleList()
        if self.bidirectional:
            self.backward_layers = nn.ModuleList()

        lstm_embed_dim = embed_dim
        for _ in range(self.num_layers):
            self.forward_layers.append(
                AugmentedLSTMUnidirectional(
                    lstm_embed_dim,
                    self.lstm_dim,
                    go_forward=True,
                    recurrent_dropout_probability=self.dropout,
                    use_highway=self.use_highway,
                    use_input_projection_bias=self.use_bias,
                )
            )
            if self.bidirectional:
                self.backward_layers.append(
                    AugmentedLSTMUnidirectional(
                        lstm_embed_dim,
                        self.lstm_dim,
                        go_forward=False,
                        recurrent_dropout_probability=self.dropout,
                        use_highway=self.use_highway,
                        use_input_projection_bias=self.use_bias,
                    )
                )

            lstm_embed_dim = self.lstm_dim * num_directions
        self.representation_dim = lstm_embed_dim
        log_class_usage(__class__)

    def forward(
        self,
        embedded_tokens: torch.Tensor,
        seq_lengths: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Given an input batch of sequential data such as word embeddings, produces
        a AugmentedLSTM representation of the sequential input and new state
        tensors.

        Args:
            embedded_tokens (torch.Tensor): Input tensor of shape
                (bsize x seq_len x input_dim).
            seq_lengths (torch.Tensor): List of sequences lengths of each batch element.
            states (Tuple[torch.Tensor, torch.Tensor]): Tuple of tensors containing
                the initial hidden state and the cell state of each element in
                the batch. Each of these tensors have a dimension of
                (bsize x num_layers x num_directions * nhid). Defaults to `None`.

        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                AgumentedLSTM representation of input and
                the state of the LSTM `t = seq_len`.
                Shape of representation is (bsize x seq_len x representation_dim).
                Shape of each state is (bsize x num_layers * num_directions x nhid).

        """
        rnn_input = pack_padded_sequence(
            embedded_tokens, seq_lengths.int(), batch_first=True
        )
        if states is not None:
            states = (states[0].transpose(0, 1), states[1].transpose(0, 1))
        if self.bidirectional:
            return self._forward_bidirectional(rnn_input, states)
        return self._forward_unidirectional(rnn_input, states)

    def _forward_bidirectional(
        self,
        inputs: PackedSequence,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ):
        output_sequence = inputs
        final_h = []
        final_c = []

        if not states:
            hidden_states = [None] * self.num_layers
        elif states[0].size()[0] != self.num_layers:
            raise RuntimeError(
                "Initial states were passed to forward() but the number of "
                "initial states does not match the number of layers."
            )
        else:
            hidden_states = list(  # noqa
                zip(
                    states[0].chunk(self.num_layers, 0),
                    states[1].chunk(self.num_layers, 0),
                )
            )
        for i, state in enumerate(hidden_states):
            if state:
                forward_state = state[0].chunk(2, -1)
                backward_state = state[1].chunk(2, -1)
            else:
                forward_state = backward_state = None

            forward_layer = self.forward_layers[i]
            backward_layer = self.backward_layers[i]
            # The state is duplicated to mirror the Pytorch API for LSTMs.
            forward_output, final_forward_state = forward_layer(
                output_sequence, forward_state
            )
            backward_output, final_backward_state = backward_layer(
                output_sequence, backward_state
            )
            forward_output, lengths = pad_packed_sequence(
                forward_output, batch_first=True
            )
            backward_output, _ = pad_packed_sequence(backward_output, batch_first=True)
            output_sequence = torch.cat([forward_output, backward_output], -1)
            output_sequence = pack_padded_sequence(
                output_sequence, lengths, batch_first=True
            )

            final_h.extend([final_forward_state[0], final_backward_state[0]])
            final_c.extend([final_forward_state[1], final_backward_state[1]])

        final_h = torch.cat(final_h, dim=0).transpose(0, 1)
        final_c = torch.cat(final_c, dim=0).transpose(0, 1)
        final_state_tuple = (final_h, final_c)
        output_sequence, _ = pad_packed_sequence(
            output_sequence, padding_value=self.padding_value, batch_first=True
        )
        return output_sequence, final_state_tuple

    def _forward_unidirectional(
        self,
        inputs: PackedSequence,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ):
        output_sequence = inputs
        final_h = []
        final_c = []

        if not states:
            hidden_states = [None] * self.num_layers
        elif states[0].size()[0] != self.num_layers:
            raise RuntimeError(
                "Initial states were passed to forward() but the number of "
                "initial states does not match the number of layers."
            )
        else:
            hidden_states = list(  # noqa
                zip(
                    states[0].chunk(self.num_layers, 0),
                    states[1].chunk(self.num_layers, 0),
                )
            )

        for i, state in enumerate(hidden_states):
            forward_layer = self.forward_layers[i]
            # The state is duplicated to mirror the Pytorch API for LSTMs.
            forward_output, final_forward_state = forward_layer(output_sequence, state)
            output_sequence = forward_output
            final_h.append(final_forward_state[0])
            final_c.append(final_forward_state[1])

        final_h = torch.cat(final_h, dim=0).transpose(0, 1)
        final_c = torch.cat(final_c, dim=0).transpose(0, 1)
        final_state_tuple = (final_h, final_c)
        output_sequence, _ = pad_packed_sequence(
            output_sequence, padding_value=self.padding_value, batch_first=True
        )
        return output_sequence, final_state_tuple

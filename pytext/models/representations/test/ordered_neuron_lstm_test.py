#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import unittest

import torch
from pytext.models.representations.ordered_neuron_lstm import OrderedNeuronLSTM


class OrderedNeuronLSTMTest(unittest.TestCase):
    def _test_shape(self, dropout, num_layers):
        config = OrderedNeuronLSTM.Config()
        config.dropout = dropout
        config.num_layers = num_layers

        batch_size = 3
        time_step = 17
        input_size = 31
        lstm = OrderedNeuronLSTM(config, input_size)

        input_tensor = torch.randn(batch_size, time_step, input_size)
        input_length = torch.zeros((batch_size,)).long()
        input_states = (
            torch.randn(batch_size, config.num_layers, config.lstm_dim),
            torch.randn(batch_size, config.num_layers, config.lstm_dim),
        )

        for i in range(batch_size):
            input_length[i] = time_step - i

        for inp_state in [None, input_states]:
            output, (hidden_state, cell_state) = lstm(
                input_tensor, input_length, inp_state
            )

            # Test Shapes
            self.assertEqual(
                hidden_state.size(), (batch_size, config.num_layers, config.lstm_dim)
            )
            self.assertEqual(
                hidden_state.size(), (batch_size, config.num_layers, config.lstm_dim)
            )
            self.assertEqual(
                cell_state.size(), (batch_size, config.num_layers, config.lstm_dim)
            )

            # Make sure gradients propagate correctly
            output_agg = output.sum()
            output_agg.backward()
            for param in lstm.parameters():
                self.assertEqual(torch.isnan(param).long().sum(), 0)
                self.assertEqual(torch.isinf(param).long().sum(), 0)

            # Make sure dropout actually does something
            s_output, (s_hidden_state, s_cell_state) = lstm(
                input_tensor, input_length, inp_state
            )

            if config.dropout == 0.0:
                assert torch.all(
                    torch.lt(torch.abs(torch.add(s_output, -output)), 1e-12)
                )
            else:
                assert not torch.all(torch.eq(s_output, output))

    def test_ordered_neuron_lstm(self):
        # test every configuration
        for num_layers in [1, 2, 3]:
            for dropout in [0.0, 0.5]:
                self._test_shape(dropout, num_layers)


if __name__ == "__main__":
    unittest.main()

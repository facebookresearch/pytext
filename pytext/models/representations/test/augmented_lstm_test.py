#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import unittest

import torch
from pytext.models.representations.augmented_lstm import AugmentedLSTM


class AugmentedLSTMTest(unittest.TestCase):
    def _test_shape(self, use_highway, variational_dropout, num_layers, bidirectional):
        config = AugmentedLSTM.Config()
        config.use_highway = use_highway
        config.dropout = variational_dropout
        config.num_layers = num_layers
        config.bidirectional = bidirectional

        num_directions = int(bidirectional) + 1

        batch_size = 3
        time_step = 17
        embed_dim = 31

        aug_lstm = AugmentedLSTM(config, embed_dim)

        input_tensor = torch.randn(batch_size, time_step, embed_dim)
        input_length = torch.zeros((batch_size,)).long()
        input_states = (
            torch.randn(
                batch_size, config.num_layers, config.lstm_dim * num_directions
            ),
            torch.randn(
                batch_size, config.num_layers, config.lstm_dim * num_directions
            ),
        )
        for i in range(batch_size):
            input_length[i] = time_step - i

        for inp_state in [None, input_states]:
            output, (hidden_state, cell_state) = aug_lstm(
                input_tensor, input_length, inp_state
            )

            # Test Shapes
            self.assertEqual(
                output.size(), (batch_size, time_step, config.lstm_dim * num_directions)
            )
            self.assertEqual(
                hidden_state.size(),
                (batch_size, config.num_layers * num_directions, config.lstm_dim),
            )
            self.assertEqual(
                cell_state.size(),
                (batch_size, config.num_layers * num_directions, config.lstm_dim),
            )

            # Make sure gradients propagate correctly
            output_agg = output.sum()
            output_agg.backward()
            for param in aug_lstm.parameters():
                self.assertEqual(torch.isnan(param).long().sum(), 0)
                self.assertEqual(torch.isinf(param).long().sum(), 0)

            # Make sure dropout actually does something
            s_output, (s_hidden_state, s_cell_state) = aug_lstm(
                input_tensor, input_length, inp_state
            )
            if config.dropout == 0.0:
                assert torch.all(
                    torch.lt(torch.abs(torch.add(s_output, -output)), 1e-12)
                )
            else:
                assert not torch.all(torch.eq(s_output, output))

    def test_augmented_lstm(self):
        # test every configuration
        for num_layers in [1, 2, 3]:
            for dropout in [0.0, 0.5]:
                for use_highway in [True, False]:
                    for bi in [True, False]:
                        self._test_shape(use_highway, dropout, num_layers, bi)


if __name__ == "__main__":
    unittest.main()

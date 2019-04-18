#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import unittest

import torch
from pytext.models.representations.augmented_lstm import AugmentedLSTM
from pytext.models.representations.bilstm import BiLSTM
from pytext.models.representations.rnn_union import RNNUnion


class RNNUnionTest(unittest.TestCase):
    def test_dispatching_auglstm(self):
        auglstm_config = RNNUnion.Config()
        auglstm_config.rnn_type = "auglstm"

        auglstm = RNNUnion(auglstm_config, 32)
        self.assertTrue(isinstance(auglstm.inner_rnn, AugmentedLSTM))

    def test_dispatching_bilstm(self):
        bilstm_config = RNNUnion.Config()
        bilstm_config.rnn_type = "bilstm"

        bilstm = RNNUnion(bilstm_config, 32)
        self.assertTrue(isinstance(bilstm.inner_rnn, BiLSTM))

    def test_dispatching_not_supported(self):
        with self.assertRaises(ValueError) as context:
            broken_config = RNNUnion.Config()
            broken_config.rnn_type = "some_rnn_cell_that_will_never_exist"
            broken = RNNUnion(broken_config, 32)
            # should raise error... but to get rid of error we use broken value
            broken
            context


if __name__ == "__main__":
    unittest.main()

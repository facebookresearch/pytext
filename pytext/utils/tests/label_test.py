#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

import numpy as np
from pytext.utils import label


class LabelUtilTest(unittest.TestCase):
    def test_get_label_weights(self):
        vocab = {"foo": 0, "bar": 1}
        weights = {"foo": 3.2, "foobar": 2.1}
        weights_tensor = label.get_label_weights(vocab, weights)
        np.testing.assert_array_almost_equal(
            np.array([3.2, 1]), weights_tensor.detach().numpy()
        )

    def test_get_auto_label_weights(self):
        vocab_dict = {"foo": 0, "bar": 1}
        label_counts = {"foo": 4, "bar": 1}
        weights_tensor = label.get_auto_label_weights(vocab_dict, label_counts)
        np.testing.assert_array_almost_equal(
            np.array([0.25, 4]), weights_tensor[0].detach().numpy()
        )

    def test_get_normalized_sqrt_label_weights(self):
        vocab_dict = {"foo": 0, "bar": 1}
        label_counts = {"foo": 4, "bar": 1}
        weights_tensor = label.get_normalized_sqrt_label_weights(
            vocab_dict, label_counts
        )
        np.testing.assert_array_almost_equal(
            np.array([0.5, 2]), weights_tensor[0].detach().numpy()
        )

    def test_get_normalized_cap_label_weights(self):
        vocab_dict = {"foo": 0, "bar": 1}
        label_counts = {"foo": 4, "bar": 1}
        weights_tensor = label.get_normalized_cap_label_weights(
            vocab_dict, label_counts
        )
        np.testing.assert_array_almost_equal(
            np.array([0.625, 1]), weights_tensor[0].detach().numpy()
        )

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

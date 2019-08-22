#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import numpy as np
import torch
from pytext.utils.cuda import Variable
from pytext.utils.model import to_onehot


class ModelUtilsTest(unittest.TestCase):
    def test_to_onehot(self):
        feat_vec = Variable(torch.LongTensor([[0, 1, 2], [3, 4, 0]]))
        onehot = to_onehot(feat_vec, 5)
        self.assertEqual(onehot.size()[0], 2)
        self.assertEqual(onehot.size()[1], 3)
        self.assertEqual(onehot.size()[2], 5)

        expected = np.zeros((2, 3, 5))
        expected[0][0][0] = 1
        expected[0][1][1] = 1
        expected[0][2][2] = 1
        expected[1][0][3] = 1
        expected[1][1][4] = 1
        expected[1][2][0] = 1

        for (i, row) in enumerate(onehot):
            for (j, feat) in enumerate(row):
                for (k, val) in enumerate(feat):
                    self.assertEqual(expected[i][j][k], val.item())

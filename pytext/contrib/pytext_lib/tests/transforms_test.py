#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reservedimport unittest

import unittest

import torch
from pytext.contrib.pytext_lib.transforms import TruncateTransform


class TestTruncateTranform(unittest.TestCase):

    DATA = [[0, 1, 2, 3, 4, 5, 6], [0, 1, 2], []]
    MAX_SEQ_LEN = 4

    def test_truncate_transform(self):
        transform = TruncateTransform(max_seq_len=TestTruncateTranform.MAX_SEQ_LEN)
        res = transform(TestTruncateTranform.DATA)

        for row in res:
            # Truncate lengths above max_seq_len, smaller lens aren't padded.
            self.assertEqual(len(row), min(TestTruncateTranform.MAX_SEQ_LEN, len(row)))

    def test_truncate_transform_torchscript(self):
        transform = TruncateTransform(max_seq_len=TestTruncateTranform.MAX_SEQ_LEN)
        ts_transform = torch.jit.script(transform)
        res = ts_transform(TestTruncateTranform.DATA)

        for row in res:
            # Truncate lengths above max_seq_len, smaller lens aren't padded.
            self.assertEqual(len(row), min(TestTruncateTranform.MAX_SEQ_LEN, len(row)))

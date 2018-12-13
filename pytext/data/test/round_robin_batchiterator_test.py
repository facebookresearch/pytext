#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

from pytext.data.disjoint_multitask_data_handler import RoundRobinBatchIterator


class RoundRobinBatchIteratorTest(unittest.TestCase):
    def test_batch_iterator(self):
        iteratorA = [(input, None, {}) for input in [1, 2, 3, 4]]
        iteratorB = [(input, None, {}) for input in ["a", "b", "c"]]
        round_robin_iterator = RoundRobinBatchIterator(
            {"A": iteratorA, "B": iteratorB}, epoch_size=10
        )
        expected_output = [1, "a", 2, "b", 3, "c", 4, "a", 1, "b"]
        for actual, expected in zip(round_robin_iterator, expected_output):
            assert actual[0] == expected

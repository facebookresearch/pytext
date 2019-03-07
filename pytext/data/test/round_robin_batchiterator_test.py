#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

from pytext.data.disjoint_multitask_data_handler import RoundRobinBatchIterator


class RoundRobinBatchIteratorTest(unittest.TestCase):
    def test_batch_iterator(self):
        iteratorA = [(input, None, {}) for input in ["1", "2", "3", "4", "5"]]
        iteratorB = [(input, None, {}) for input in ["a", "b", "c"]]

        # upsample = True, no iter_to_set_epoch
        round_robin_iterator = RoundRobinBatchIterator(
            {"A": iteratorA, "B": iteratorB}, upsample=True
        )
        expected_items = ["1", "a", "2", "b", "3", "c"]
        self._check_iterator(round_robin_iterator, expected_items)

        # upsample = True, iter_to_set_epoch = "A"
        round_robin_iterator = RoundRobinBatchIterator(
            {"A": iteratorA, "B": iteratorB}, upsample=True, iter_to_set_epoch="A"
        )
        expected_items = ["1", "a", "2", "b", "3", "c", "4", "a", "5", "b"]
        self._check_iterator(round_robin_iterator, expected_items)

        # upsample = False
        round_robin_iterator = RoundRobinBatchIterator(
            {"A": iteratorA, "B": iteratorB}, upsample=False
        )
        expected_items = ["1", "2", "3", "4", "5", "a", "b", "c"]
        self._check_iterator(round_robin_iterator, expected_items, fixed_order=False)

    def _check_iterator(self, iterator, expected_items, fixed_order=True):
        actual_items = [item for item, _, _ in iterator]
        if not fixed_order:
            # Order is random, just check that the sorted arrays are equal
            actual_items = sorted(actual_items)
            expected_items = sorted(expected_items)
        self.assertListEqual(actual_items, expected_items)

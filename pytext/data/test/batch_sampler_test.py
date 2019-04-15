#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

from pytext.data import EvalBatchSampler, RoundRobinBatchSampler


class BatchSamplerTest(unittest.TestCase):
    def test_batch_sampler(self):
        iteratorA = ["1", "2", "3", "4", "5"]
        iteratorB = ["a", "b", "c"]

        # no iter_to_set_epoch
        round_robin_iterator = RoundRobinBatchSampler().batchify(
            {"A": iteratorA, "B": iteratorB}
        )
        expected_items = ["1", "a", "2", "b", "3", "c", "4"]
        self._check_iterator(round_robin_iterator, expected_items)

        # iter_to_set_epoch = "A"
        round_robin_iterator = RoundRobinBatchSampler(iter_to_set_epoch="A").batchify(
            {"A": iteratorA, "B": iteratorB}
        )
        expected_items = ["1", "a", "2", "b", "3", "c", "4", "a", "5", "b"]
        self._check_iterator(round_robin_iterator, expected_items)

        eval_iterator = EvalBatchSampler().batchify({"A": iteratorA, "B": iteratorB})
        expected_items = ["1", "2", "3", "4", "5", "a", "b", "c"]
        self._check_iterator(eval_iterator, expected_items)

    def _check_iterator(self, iterator, expected_items, fixed_order=True):
        actual_items = [item for _, item in iterator]
        if not fixed_order:
            # Order is random, just check that the sorted arrays are equal
            actual_items = sorted(actual_items)
            expected_items = sorted(expected_items)
        self.assertListEqual(actual_items, expected_items)

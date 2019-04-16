#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

from pytext.data import (
    EvalBatchSampler,
    ProbabalisticBatchSampler,
    RoundRobinBatchSampler,
)


class BatchSamplerTest(unittest.TestCase):
    def setUp(self):
        self.iteratorA = ["1", "2", "3", "4", "5"]
        self.iteratorB = ["a", "b", "c"]
        self.iter_dict = {"A": self.iteratorA, "B": self.iteratorB}

    def test_round_robin_batch_sampler(self):

        # no iter_to_set_epoch
        round_robin_iterator = RoundRobinBatchSampler().batchify(self.iter_dict)
        expected_items = ["1", "a", "2", "b", "3", "c", "4"]
        self._check_iterator(round_robin_iterator, expected_items)

        # iter_to_set_epoch = "A"
        round_robin_iterator = RoundRobinBatchSampler(iter_to_set_epoch="A").batchify(
            self.iter_dict
        )
        expected_items = ["1", "a", "2", "b", "3", "c", "4", "a", "5", "b"]
        self._check_iterator(round_robin_iterator, expected_items)

    def test_eval_batch_sampler(self):
        eval_iterator = EvalBatchSampler().batchify(self.iter_dict)
        expected_items = ["1", "2", "3", "4", "5", "a", "b", "c"]
        self._check_iterator(eval_iterator, expected_items)

    def test_prob_batch_sampler(self):
        prob_iterator = ProbabalisticBatchSampler(
            iterator_probabilities={"A": 0, "B": 1}, epoch_size=-1
        ).batchify(self.iter_dict)
        expected_items = ["a", "b", "c"]
        self._check_iterator(prob_iterator, expected_items)

        prob_iterator = ProbabalisticBatchSampler(
            iterator_probabilities={"A": 1, "B": 0}, epoch_size=-1
        ).batchify(self.iter_dict)
        expected_items = ["1", "2", "3", "4", "5"]
        self._check_iterator(prob_iterator, expected_items)

        prob_iterator = ProbabalisticBatchSampler(
            iterator_probabilities={"A": 1, "B": 0}, epoch_size=10
        ).batchify(self.iter_dict)
        expected_items = ["1", "2", "3", "4", "5", "1", "2", "3", "4", "5"]
        self._check_iterator(prob_iterator, expected_items)

    def _check_iterator(self, iterator, expected_items, fixed_order=True):
        actual_items = [item for _, item in iterator]
        if not fixed_order:
            # Order is random, just check that the sorted arrays are equal
            actual_items = sorted(actual_items)
            expected_items = sorted(expected_items)
        self.assertListEqual(actual_items, expected_items)

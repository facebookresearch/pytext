#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

from pytext.data import (
    AlternatingRandomizedBatchSampler,
    EvalBatchSampler,
    RandomizedBatchSampler,
    RoundRobinBatchSampler,
)


class BatchSamplerTest(unittest.TestCase):
    def setUp(self):
        self.iteratorA = ["1", "2", "3", "4", "5"]
        self.iteratorB = ["a", "b", "c"]
        self.iter_dict = {"A": self.iteratorA, "B": self.iteratorB}

        self.iteratorC = ["d", "e", "f"]
        self.iteratorD = ["6", "7", "8", "9", "10"]
        self.alternating_iter_dict = {
            "A": self.iteratorA,
            "B": self.iteratorB,
            "C": self.iteratorC,
            "D": self.iteratorD,
        }

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
        sampler = RandomizedBatchSampler(unnormalized_iterator_probs={"A": 1, "B": 0})

        prob_iterator = self._truncate(iter(sampler.batchify(self.iter_dict)), 8)
        expected_items = ["1", "2", "3", "4", "5", "1", "2", "3"]
        self._check_iterator(prob_iterator, expected_items)
        prob_iterator = self._truncate(iter(sampler.batchify(self.iter_dict)), 8)
        expected_items = ["4", "5", "1", "2", "3", "4", "5", "1"]
        self._check_iterator(prob_iterator, expected_items)

        sampler = RandomizedBatchSampler(unnormalized_iterator_probs={"A": 0, "B": 1})
        prob_iterator = self._truncate(sampler.batchify(self.iter_dict), 5)
        expected_items = ["a", "b", "c", "a", "b"]
        self._check_iterator(prob_iterator, expected_items)
        prob_iterator = self._truncate(sampler.batchify(self.iter_dict), 5)
        expected_items = ["c", "a", "b", "c", "a"]
        self._check_iterator(prob_iterator, expected_items)

    def test_alternate_prob_batch_sampler(self):
        sampler = AlternatingRandomizedBatchSampler(
            unnormalized_iterator_probs={"A": 1, "B": 0},
            second_unnormalized_iterator_probs={"C": 0, "D": 1},
        )

        prob_iterator = self._truncate(
            iter(sampler.batchify(self.alternating_iter_dict)), 12
        )
        expected_items = ["1", "6", "2", "7", "3", "8", "4", "9", "5", "10", "1", "6"]
        self._check_iterator(prob_iterator, expected_items)
        prob_iterator = self._truncate(
            iter(sampler.batchify(self.alternating_iter_dict)), 4
        )
        expected_items = ["2", "7", "3", "8"]
        self._check_iterator(prob_iterator, expected_items)

        sampler = AlternatingRandomizedBatchSampler(
            unnormalized_iterator_probs={"A": 0, "B": 1},
            second_unnormalized_iterator_probs={"C": 1, "D": 0},
        )
        prob_iterator = self._truncate(sampler.batchify(self.alternating_iter_dict), 9)
        expected_items = ["a", "d", "b", "e", "c", "f", "a", "d", "b"]
        self._check_iterator(prob_iterator, expected_items)
        prob_iterator = self._truncate(sampler.batchify(self.alternating_iter_dict), 3)
        expected_items = ["e", "c", "f"]
        self._check_iterator(prob_iterator, expected_items)

    def _check_iterator(self, iterator, expected_items, fixed_order=True):
        actual_items = [item for _, item in iterator]
        if not fixed_order:
            # Order is random, just check that the sorted arrays are equal
            actual_items = sorted(actual_items)
            expected_items = sorted(expected_items)
        self.assertListEqual(actual_items, expected_items)

    def _truncate(self, items, length):
        for _ in range(length):
            yield next(items)

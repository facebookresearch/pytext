#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest
from typing import List

from pytext.data.data import BatchData
from pytext.data.sources import RawExample
from pytext.data.token_batcher import TokenBatcher
from pytext.utils.test import import_tests_module


tests_module = import_tests_module()


class TokenBatcherTest(unittest.TestCase):
    @classmethod
    def _get_dataset(cls, dataset_size: int, num_tokens: int) -> List[RawExample]:
        return [
            BatchData(
                {"source_sequence": ("word " * num_tokens)},
                {"a": i, "b": 10 + i, "c": 20 + i},
            )
            for i in range(dataset_size)
        ]

    def test_various_token_batcher(self):
        # test most basic case
        self.test_token_batcher_helper(
            dataset_size=100,
            num_tokens=1,
            bsz_mult=1,
            max_tokens=10,
            output_one_length=10,
            output_two_length=10,
        )

        # test running through entire dataset
        self.test_token_batcher_helper(
            dataset_size=100,
            num_tokens=1,
            bsz_mult=1,
            max_tokens=55,
            output_one_length=55,
            output_two_length=45,
        )

        # test when max_tokens doesn't exactly divide num_tokens
        self.test_token_batcher_helper(
            dataset_size=20,
            num_tokens=3,
            bsz_mult=1,
            max_tokens=20,
            output_one_length=6,
            output_two_length=6,
        )

        # test variation on bsz_mult
        self.test_token_batcher_helper(
            dataset_size=20,
            num_tokens=3,
            bsz_mult=2,
            max_tokens=20,
            output_one_length=6,
            output_two_length=6,
        )

    def test_token_batcher_helper(
        self,
        dataset_size: int = 100,
        num_tokens: int = 1,
        bsz_mult: int = 1,
        max_tokens: int = 10,
        output_one_length: int = 10,
        output_two_length: int = 10,
    ):
        data = self._get_dataset(dataset_size, num_tokens)

        batcher_config = TokenBatcher.Config(
            train_batch_size=1,
            eval_batch_size=1,
            test_batch_size=1,
            pool_num_batches=1,
            num_shuffled_pools=1,
            bsz_mult=bsz_mult,
            max_tokens=max_tokens,
        )

        batcher = TokenBatcher.from_config(batcher_config)

        # # epoch 1
        batches = list(batcher.batchify(data))
        self.assertEqual(len(batches[0][0]), output_one_length)

        # # epoch 2
        self.assertEqual(len(batches[1][0]), output_two_length)

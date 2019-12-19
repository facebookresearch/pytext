#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest
from typing import List

from pytext.data.dynamic_pooling_batcher import (
    BatcherSchedulerConfig,
    DynamicPoolingBatcher,
    ExponentialBatcherSchedulerConfig,
    ExponentialDynamicPoolingBatcher,
    LinearDynamicPoolingBatcher,
)
from pytext.data.sources import RawExample
from pytext.utils.test import import_tests_module


tests_module = import_tests_module()


class DynamicPoolingBatcherTest(unittest.TestCase):
    @classmethod
    def _get_dataset(cls, dataset_size: int) -> List[RawExample]:
        return [([], {})] * dataset_size

    def test_linear_scheduler(self):
        data = DynamicPoolingBatcherTest._get_dataset(dataset_size=100)

        batch_scheduler_config = BatcherSchedulerConfig(
            start_batch_size=32, end_batch_size=256, epoch_period=5, step_size=1
        )

        batcher_config = DynamicPoolingBatcher.Config(
            train_batch_size=1,
            eval_batch_size=1,
            test_batch_size=1,
            pool_num_batches=1,
            num_shuffled_pools=1,
            scheduler_config=batch_scheduler_config,
        )

        batcher = LinearDynamicPoolingBatcher.from_config(batcher_config)

        # epoch 1
        batches = [item for item in batcher.batchify(data)]
        self.assertEqual(len(batches[0].raw_data), 32)

        # epoch 2
        # new size ()(256-32) / 5) + 32 = 76.8 ~ 77
        batches = [item for item in batcher.batchify(data)]
        self.assertEqual(len(batches[0].raw_data), 77)

    def test_exponential_scheduler(self):
        data = DynamicPoolingBatcherTest._get_dataset(dataset_size=100)

        batch_scheduler_config = ExponentialBatcherSchedulerConfig(
            start_batch_size=32,
            end_batch_size=256,
            epoch_period=5,
            step_size=1,
            gamma=2,
        )

        batcher_config = ExponentialDynamicPoolingBatcher.Config(
            train_batch_size=1,
            eval_batch_size=1,
            test_batch_size=1,
            pool_num_batches=1,
            num_shuffled_pools=1,
            scheduler_config=batch_scheduler_config,
        )

        batcher = ExponentialDynamicPoolingBatcher.from_config(batcher_config)

        # epoch 1
        batches = [item for item in batcher.batchify(data)]
        self.assertEqual(len(batches[0].raw_data), 32)

        # epoch 2
        # new size 32 * 2^1 = 64
        batches = [item for item in batcher.batchify(data)]
        self.assertEqual(len(batches[0].raw_data), 64)

    def test_batch_size_greater_than_data(self):
        data = DynamicPoolingBatcherTest._get_dataset(dataset_size=50)

        batch_scheduler_config = ExponentialBatcherSchedulerConfig(
            start_batch_size=32,
            end_batch_size=256,
            epoch_period=5,
            step_size=1,
            gamma=2,
        )

        batcher_config = ExponentialDynamicPoolingBatcher.Config(
            train_batch_size=1,
            eval_batch_size=1,
            test_batch_size=1,
            pool_num_batches=1,
            num_shuffled_pools=1,
            scheduler_config=batch_scheduler_config,
        )

        batcher = ExponentialDynamicPoolingBatcher.from_config(batcher_config)

        # epoch 1
        batches = [item for item in batcher.batchify(data)]
        self.assertEqual(len(batches[0].raw_data), 32)

        # epoch 2
        # new size 32 * 2^1 = 64 / 8 = 8
        batches = [item for item in batcher.batchify(data)]
        self.assertEqual(len(batches[0].raw_data), 50)

    def end_of_scheduler(self):
        data = DynamicPoolingBatcherTest._get_dataset(dataset_size=300)

        batch_scheduler_config = ExponentialBatcherSchedulerConfig(
            start_batch_size=32,
            end_batch_size=256,
            epoch_period=2,
            step_size=4,
            gamma=2,
        )

        batcher_config = ExponentialDynamicPoolingBatcher.Config(
            train_batch_size=1,
            eval_batch_size=1,
            test_batch_size=1,
            pool_num_batches=1,
            num_shuffled_pools=1,
            scheduler_config=batch_scheduler_config,
        )

        batcher = ExponentialDynamicPoolingBatcher.from_config(batcher_config)

        # epoch 1
        batches = [item for item in batcher.batchify(data)]
        self.assertEqual(len(batches[0].raw_data), 32)

        # pass N epochs
        no_op_epochs = 4
        _ = [[item for item in batcher.batchify(data)] for _ in range(no_op_epochs)]

        # after period is passed, batch size should be max batch size
        batches = [item for item in batcher.batchify(data)]
        self.assertEqual(len(batches[0].raw_data), 256)

    def test_step_size(self):
        data = DynamicPoolingBatcherTest._get_dataset(dataset_size=64)

        batch_scheduler_config = ExponentialBatcherSchedulerConfig(
            start_batch_size=32,
            end_batch_size=256,
            epoch_period=2,
            step_size=2,
            gamma=2,
        )

        batcher_config = ExponentialDynamicPoolingBatcher.Config(
            train_batch_size=1,
            eval_batch_size=1,
            test_batch_size=1,
            pool_num_batches=1,
            num_shuffled_pools=1,
            scheduler_config=batch_scheduler_config,
        )

        batcher = ExponentialDynamicPoolingBatcher.from_config(batcher_config)

        # epoch 1
        batches = [item for item in batcher.batchify(data)]
        self.assertEqual(len(batches[0].raw_data), 32)

        # epoch 2
        # no op on batch size
        batches = [item for item in batcher.batchify(data)]
        self.assertEqual(len(batches[0].raw_data), 32)

        # epoch 3
        batches = [item for item in batcher.batchify(data)]
        self.assertEqual(len(batches[0].raw_data), 64)

        # epoch 4
        # no op on batch size
        batches = [item for item in batcher.batchify(data)]
        self.assertEqual(len(batches[0].raw_data), 64)

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import math
from typing import Iterable

from pytext.common.constants import Stage
from pytext.config.module_config import ModuleConfig

from .data import PoolingBatcher
from .sources import RawExample


class BatcherSchedulerConfig(ModuleConfig):
    # the initial batch size used for training, this is per node
    start_batch_size: int = 32

    # the final or max batch size to use, any scheduler should
    # not go over this batch size, this is per node
    end_batch_size: int = 256

    # the number of epochs to increase the batch size over
    epoch_period: int = 10

    # the batch size is kept constant for `step_size` number of epochs
    step_size: int = 1


class ExponentialBatcherSchedulerConfig(BatcherSchedulerConfig):
    # the gamma to increase batch size by
    gamma: float = 5


class DynamicPoolingBatcher(PoolingBatcher):
    """
    Allows dynamic batch training, extends pooling batcher with a scheduler
    config, which specifies how batch size should increase
    """

    class Config(PoolingBatcher.Config):
        scheduler_config: BatcherSchedulerConfig = BatcherSchedulerConfig()

    @classmethod
    def from_config(cls, config: Config):
        return cls(
            config.train_batch_size,
            config.eval_batch_size,
            config.test_batch_size,
            config.pool_num_batches,
            config.num_shuffled_pools,
            config.scheduler_config,
        )

    def __init__(
        self,
        train_batch_size=Config.train_batch_size,
        eval_batch_size=Config.eval_batch_size,
        test_batch_size=Config.test_batch_size,
        pool_num_batches=Config.pool_num_batches,
        num_shuffled_pools=Config.num_shuffled_pools,
        scheduler_config=Config.scheduler_config,
    ):
        super().__init__(
            train_batch_size,
            eval_batch_size,
            test_batch_size,
            pool_num_batches,
            num_shuffled_pools,
        )
        self.scheduler_config = scheduler_config
        self.curr_batch_size: int = 0
        self.curr_epoch: int = -1
        self.step_epoch()

    def compute_dynamic_batch_size(
        self, curr_epoch: int, scheduler_config: BatcherSchedulerConfig, curr_steps: int
    ) -> int:
        raise NotImplementedError()

    def step_epoch(self):
        self.curr_epoch += 1
        if self.curr_epoch > self.scheduler_config.epoch_period:
            # if past the dynamic period just return
            # the batch size
            self.curr_batch_size = self.scheduler_config.end_batch_size
        else:
            if self.curr_epoch % self.scheduler_config.step_size == 0:
                old_size = self.curr_batch_size
                new_size = self.compute_dynamic_batch_size(
                    curr_epoch=self.curr_epoch,
                    scheduler_config=self.scheduler_config,
                    curr_steps=self.curr_epoch // self.scheduler_config.step_size,
                )

                print(f"increasing batch size from {old_size} to {new_size}")

                self.curr_batch_size = new_size

    def finished_dynamic(self) -> bool:
        return (
            self.scheduler_config.epoch_period != -1
            and self.curr_epoch >= self.scheduler_config.epoch_period
        )

    def get_batch_size(self, stage: Stage) -> int:
        if stage == Stage.TRAIN:
            print(f"using dynamic batch size {self.curr_batch_size}")
            return self.curr_batch_size
        else:
            return self._batch_sizes[stage]

    def batchify(
        self, iterable: Iterable[RawExample], sort_key=None, stage=Stage.TRAIN
    ):
        """
        From an iterable of dicts, yield dicts of lists:

        1. Load `num_shuffled_pools` pools of data, and shuffle them.
        2. Load a pool (`batch_size * pool_num_batches` examples).
        3. Sort rows, if necessary.
        4. Shuffle the order in which the batches are returned, if necessary.
        """
        for item in super().batchify(iterable=iterable, sort_key=sort_key, stage=stage):
            yield item
        if stage == Stage.TRAIN:
            # only step scheduler when in train
            self.step_epoch()


class LinearDynamicPoolingBatcher(DynamicPoolingBatcher):
    """
    Linear Dynamic Batch Scheduler: scales up batch size linearly
    """

    def compute_dynamic_batch_size(
        self, curr_epoch: int, scheduler_config: BatcherSchedulerConfig, curr_steps: int
    ) -> int:
        batch_delta = (
            scheduler_config.end_batch_size - scheduler_config.start_batch_size
        )
        curr_est: float = (
            batch_delta / (scheduler_config.epoch_period / scheduler_config.step_size)
        ) * curr_steps + scheduler_config.start_batch_size
        return math.ceil(curr_est)


class ExponentialDynamicPoolingBatcher(DynamicPoolingBatcher):
    """
    Exponential Dynamic Batch Scheduler: scales up batch size by a factor of
    gamma
    """

    class Config(DynamicPoolingBatcher.Config):
        scheduler_config: ExponentialBatcherSchedulerConfig

    def __init__(self, *args, **kwargs):
        self.max_steps = None
        super().__init__(*args, **kwargs)

    def get_max_steps(self):
        if self.max_steps:
            return self.max_steps
        self.max_steps: float = math.floor(
            math.log(
                self.scheduler_config.end_batch_size
                / self.scheduler_config.start_batch_size
            )
            / math.log(self.scheduler_config.gamma)
        )

        return self.max_steps

    def finished_dynamic(self) -> bool:
        return (
            self.scheduler_config.epoch_period != -1
            and self.curr_epoch >= self.get_max_steps()
        )

    def compute_dynamic_batch_size(
        self,
        curr_epoch: int,
        scheduler_config: ExponentialBatcherSchedulerConfig,
        curr_steps: int,
    ) -> int:
        if curr_steps > self.get_max_steps():
            return scheduler_config.end_batch_size
        curr_est: float = scheduler_config.start_batch_size * math.pow(
            scheduler_config.gamma, curr_steps
        )
        return min(math.ceil(curr_est), scheduler_config.end_batch_size)

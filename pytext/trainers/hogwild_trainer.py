#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy

import torch
import torch.multiprocessing as mp
from pytext.common.constants import Stage
from pytext.config.pytext_config import ConfigBase
from pytext.metric_reporters import MetricReporter
from pytext.trainers.trainer import TaskTrainer, Trainer, TrainingState
from pytext.utils import cuda
from torchtext.data import Iterator


class HogwildTrainer_Deprecated(Trainer):
    class Config(ConfigBase):
        real_trainer: Trainer.Config = Trainer.Config()
        num_workers: int = 1

    @classmethod
    def from_config(cls, config: Config, model: torch.nn.Module, *args, **kwargs):
        # can't run hogwild on cuda
        if cuda.CUDA_ENABLED or config.num_workers == 1:
            return Trainer(config.real_trainer, model)
        return cls(config.real_trainer, config.num_workers, model, *args, **kwargs)

    def __init__(
        self, real_trainer_config, num_workers, model: torch.nn.Module, *args, **kwargs
    ):
        super().__init__(real_trainer_config, model, *args, **kwargs)
        self.num_workers = num_workers

    def run_epoch(
        self, state: TrainingState, data_iter: Iterator, metric_reporter: MetricReporter
    ):
        if state.stage == Stage.TRAIN:
            processes = []
            for worker_rank in range(self.num_workers):
                # Initialize the batches with different random states.
                worker_state = copy.copy(state)
                worker_state.rank = worker_rank
                data_iter.batches.init_epoch()
                p = mp.Process(
                    target=super().run_epoch, args=(state, data_iter, metric_reporter)
                )

                processes.append(p)
                p.start()
            for p in processes:
                p.join()
        else:
            return super().run_epoch(state, data_iter, metric_reporter)

    def set_up_training(self, state: TrainingState, training_data):
        training_data = super().set_up_training(state, training_data)

        # Share memory of tensors for concurrent updates from multiple processes.
        if self.num_workers > 1:
            for param in state.model.parameters():
                param.share_memory_()

        return training_data


class HogwildTrainer(TaskTrainer):
    class Config(ConfigBase):
        real_trainer: TaskTrainer.Config = TaskTrainer.Config()
        num_workers: int = 1

    @classmethod
    def from_config(cls, config: Config, model: torch.nn.Module, *args, **kwargs):
        # can't run hogwild on cuda
        if cuda.CUDA_ENABLED or config.num_workers == 1:
            return TaskTrainer(config.real_trainer, model)
        return cls(config.real_trainer, config.num_workers, model, *args, **kwargs)

    __init__ = HogwildTrainer_Deprecated.__init__
    run_epoch = HogwildTrainer_Deprecated.run_epoch
    set_up_training = HogwildTrainer_Deprecated.set_up_training

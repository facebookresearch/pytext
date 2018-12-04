#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from multiprocessing import Manager
from typing import List

import torch
import torch.multiprocessing as mp
from pytext.config import PyTextConfig
from pytext.config.component import create_trainer
from pytext.config.pytext_config import ConfigBase
from pytext.metric_reporters import MetricReporter
from pytext.models.model import Model
from pytext.trainers.trainer import Trainer
from torchtext.data import Iterator


class HogwildTrainer(Trainer):
    class Config(ConfigBase):
        real_trainer: Trainer.Config = Trainer.Config()
        num_workers: int = 1

    @classmethod
    def from_config(cls, config: Config, *args, **kwargs):
        return cls(
            create_trainer(config.real_trainer, *args, **kwargs), config.num_workers
        )

    def __init__(self, real_trainer, num_workers):
        self.real_trainer = real_trainer
        self.num_workers = num_workers

    def test(self, model, test_iter, metric_reporter):
        return self.real_trainer.test(model, test_iter, metric_reporter)

    def train(
        self,
        train_iter: Iterator,
        eval_iter: Iterator,
        model: Model,
        metric_reporter: MetricReporter,
        optimizers: List[torch.optim.Optimizer],
        pytext_config: PyTextConfig,
        scheduler=None,
        *args,
        **kwargs
    ):
        print("Num of workers for Hogwild Training is {}".format(self.num_workers))

        # Share memory of tensors for concurrent updates from multiple processes.
        if self.num_workers > 1:
            for param in model.parameters():
                param.share_memory_()

        processes = []
        for rank in range(1, self.num_workers):
            # Initialize the batches with different randome states.
            train_iter.batches.init_epoch()
            p = mp.Process(
                target=self.real_trainer.train,
                args=(
                    train_iter,
                    eval_iter,
                    model,
                    metric_reporter,
                    optimizers,
                    pytext_config,
                    scheduler,
                    None,
                    rank,
                ),
            )
            processes.append(p)
            p.start()

        training_result: List = Manager().list()  # Actual type is ListProxy.
        self.real_trainer.train(
            train_iter,
            eval_iter,
            model,
            metric_reporter,
            optimizers,
            pytext_config,
            scheduler,
            training_result,
            rank=0,
        )

        for p in processes:
            p.join()

        # Ony rank 0 worker writes to training_result
        assert len(training_result) == 1
        return training_result[0]  # Contains best model and best metric.

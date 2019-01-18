#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Any, List, Tuple

import torch
import torch.multiprocessing as mp
from pytext.common.constants import Stage
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
        return cls(config.real_trainer, config.num_workers, *args, **kwargs)

    def __init__(self, real_trainer_config, num_workers, *args, **kwargs):
        super().__init__(real_trainer_config, *args, **kwargs)
        self.num_workers = num_workers

    def _run_epoch(
        self,
        stage,
        epoch,
        data_iter,
        model,
        metric_reporter,
        pre_batch=lambda: None,
        backprop=lambda loss: None,
        rank=0,
    ):
        if stage == Stage.TRAIN:
            processes = []
            for worker_rank in range(self.num_workers):
                # Initialize the batches with different random states.
                data_iter.batches.init_epoch()
                p = mp.Process(
                    target=super()._run_epoch,
                    args=(
                        stage,
                        epoch,
                        data_iter,
                        model,
                        metric_reporter,
                        pre_batch,
                        backprop,
                        worker_rank,
                    ),
                )

                processes.append(p)
                p.start()
            for p in processes:
                p.join()
        else:
            return super()._run_epoch(
                stage,
                epoch,
                data_iter,
                model,
                metric_reporter,
                pre_batch,
                backprop,
                rank,
            )

    def train(
        self,
        train_iter: Iterator,
        eval_iter: Iterator,
        model: Model,
        metric_reporter: MetricReporter,
        optimizer: torch.optim.Optimizer,
        pytext_config: PyTextConfig,
        scheduler=None,
        *args,
        **kwargs
    ) -> Tuple[torch.nn.Module, Any]:
        print("Num of workers for Hogwild Training is {}".format(self.num_workers))

        # Share memory of tensors for concurrent updates from multiple processes.
        if self.num_workers > 1:
            for param in model.parameters():
                param.share_memory_()

        return super().train(
            train_iter,
            eval_iter,
            model,
            metric_reporter,
            optimizer,
            pytext_config,
            scheduler,
        )

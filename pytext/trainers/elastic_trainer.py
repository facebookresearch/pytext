#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import copy
import logging
from typing import Any, Optional, Tuple

import torch
import torchelastic
from pytext.config import PyTextConfig
from pytext.data.data_handler import BatchIterator
from pytext.metric_reporters import MetricReporter
from pytext.trainers.training_state import TrainingState
from pytext.utils import timing
from torchelastic.p2p import CoordinatorP2P
from torchelastic.worker_stats import WorkerStats

from .trainer import TaskTrainer, Trainer


log = logging.getLogger(__name__)


class PytextWorkerStats(WorkerStats):
    """
    ClassyVision-specific implementation of WorkerStats,
    which is used by torchelastic train_loop
    to detect (and correct stragglers), or other progress-impeding issues.
    """

    def __init__(self, progress_rate: float):
        self.progress_rate = progress_rate

    def get_progress_rate(self) -> Optional[float]:
        return self.progress_rate


class PytextElasticState(torchelastic.State):
    """
    Rollback is disabled on this state since currently, data loaders are
    too expensive to snapshot on every train_step
    """

    def __init__(self, train_state: TrainingState):
        # WARNING: Make sure to add any members here to self.save() and self.load()
        self.train_state = train_state
        self.snapshot = None

    def set_train_state(self, train_state: TrainingState):
        self.train_state = train_state

    def get_train_state(self):
        return self.train_state

    def sync(self, world_size: int, rank: int):
        """
        TODO: implement this
        """
        pass


_coordinator = None


def initialize_coordinator(rdzv_url, max_size):
    global _coordinator
    _coordinator = CoordinatorP2P(
        c10d_backend="gloo",
        init_method=rdzv_url,
        max_num_trainers=max_size,
        process_group_timeout=10000,
    )


def get_coordinator():
    global _coordinator
    assert _coordinator is not None, "coordinator not intialized."
    return _coordinator


class ElasticTrainer(TaskTrainer):
    class Config(TaskTrainer.Config):
        def __init__(self, config: Trainer.Config):
            self.__dict__ = copy.deepcopy(config.__dict__)

    @classmethod
    def from_config(cls, config: Config, model: torch.nn.Module, *args, **kwargs):
        return cls(config, model)

    def __init__(self, config: Trainer.Config, model: torch.nn.Module):
        super().__init__(config, model)

    @timing.time("Trainer.train_from_state")
    def train_from_state(
        self,
        state: TrainingState,
        training_data: BatchIterator,
        eval_data: BatchIterator,
        metric_reporter: MetricReporter,
        train_config: PyTextConfig,
    ) -> Tuple[torch.nn.Module, Any]:

        # initialize elastic state
        elastic_state = PytextElasticState(state)

        # define elastic state generator
        def elastic_state_generator(pytext_train_state):
            for pytext_train_state in self.train_from_state_internal(
                pytext_train_state.train_state,
                training_data,
                eval_data,
                metric_reporter,
                train_config,
            ):
                elastic_state.set_train_state(pytext_train_state)
                yield elastic_state, PytextWorkerStats(0)

        coordinator = get_coordinator()

        # run elastic
        elastic_state = torchelastic.run_train(
            coordinator, elastic_state_generator, elastic_state
        )
        return (
            elastic_state.train_state.model,
            elastic_state.train_state.best_model_metric,
        )

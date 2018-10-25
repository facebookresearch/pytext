#!/usr/bin/env python3

from enum import Enum
from typing import List, Optional

import torch
from pytext.config import ConfigBase
from pytext.config.component import Component, ComponentType
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ExponentialLR,
    ReduceLROnPlateau,
    StepLR,
    _LRScheduler,
)


class SchedulerType(Enum):
    NONE = "none"
    STEP_LR = "step_lr"
    EXPONENTIAL_LR = "exponential_lr"
    COSINE_ANNEALING_LR = "cosine_annealing_lr"
    REDUCE_LR_ON_PLATEAU = "reduce_lr_on_plateau"


class SchedulerParams(ConfigBase):
    """Parameters for the learning rate schedulers."""

    type: SchedulerType = SchedulerType.NONE
    step_size: int = 30
    gamma: float = 0.1
    T_max: int = 1000
    eta_min: float = 0

    # Parameters specific to `ReduceLROnPlateau` (see PyTorch docs)
    patience: int = 5
    lower_is_better: bool = True  # reduce when metric stops decreasing
    threshold: float = 0.0001
    threshold_is_absolute: bool = False  # see threshold_mode option in PyTorch
    cooldown: int = 0


class Scheduler(Component):
    """Wrapper for all schedulers.

    Wraps one of PyTorch's epoch-based learning rate schedulers or the metric-based
    `ReduceLROnPlateau`. The trainer will need to call the `step()` method at
    the end of every epoch, passing the epoch number and validation metrics.
    Note this differs slightly from PyText, where some schedulers need to be
    stepped at the beginning of each epoch.
    """

    __COMPONENT_TYPE__ = ComponentType.SCHEDULER

    Config = SchedulerParams

    def __init__(
        self, optimizers: List[torch.optim.Optimizer], scheduler_params: SchedulerParams
    ) -> None:
        self.epoch_based_schedulers: List[_LRScheduler] = []
        self.metric_based_schedulers: List[ReduceLROnPlateau] = []

        if scheduler_params.type == SchedulerType.NONE:
            pass
        elif scheduler_params.type == SchedulerType.STEP_LR:
            self.epoch_based_schedulers = [
                StepLR(optimizer, scheduler_params.step_size, scheduler_params.gamma)
                for optimizer in optimizers
            ]
        elif scheduler_params.type == SchedulerType.EXPONENTIAL_LR:
            self.epoch_based_schedulers = [
                ExponentialLR(optimizer, scheduler_params.gamma)
                for optimizer in optimizers
            ]
        elif scheduler_params.type == SchedulerType.COSINE_ANNEALING_LR:
            self.epoch_based_schedulers = [
                CosineAnnealingLR(
                    optimizer, scheduler_params.T_max, scheduler_params.eta_min
                )
                for optimizer in optimizers
            ]
        elif scheduler_params.type == SchedulerType.REDUCE_LR_ON_PLATEAU:
            self.metric_based_schedulers = [
                ReduceLROnPlateau(
                    optimizer,
                    mode="min" if scheduler_params.lower_is_better else "max",
                    factor=scheduler_params.gamma,
                    patience=scheduler_params.patience,
                    min_lr=scheduler_params.eta_min,
                    threshold=scheduler_params.threshold,
                    threshold_mode=(
                        "abs" if scheduler_params.threshold_is_absolute else "rel"
                    ),
                    cooldown=scheduler_params.cooldown,
                )
                for optimizer in optimizers
            ]
        else:
            raise ValueError("Unknown optimizer scheduler type")

    def step(self, metrics: float, epoch: Optional[int] = None) -> None:
        for epoch_based_scheduler in self.epoch_based_schedulers:
            epoch_based_scheduler.step(epoch)
        for metric_based_scheduler in self.metric_based_schedulers:
            metric_based_scheduler.step(metrics, epoch)

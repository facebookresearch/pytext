#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import math
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
    LM_FINE_TUNING_LR = "lm_fine_tuning_lr"


class SchedulerParams(ConfigBase):
    """Parameters for the learning rate schedulers."""

    type: SchedulerType = SchedulerType.NONE
    step_size: int = 30
    gamma: float = 0.1
    T_max: int = 1000
    eta_min: float = 0

    # Parameters specific to `ReduceLROnPlateau` (see PyTorch docs)
    patience: int = 5
    threshold: float = 0.0001
    threshold_is_absolute: bool = False  # see threshold_mode option in PyTorch
    cooldown: int = 0

    # Parameters specific to class `LmFineTuning` config
    cut_frac: float = 0.1
    ratio: int = 32
    non_pretrained_param_groups: int = 2  # see docstring below for default value
    lm_lr_multiplier: float = 1.0
    lm_use_per_layer_lr: bool = False
    lm_gradual_unfreezing: bool = True


class LmFineTuning(_LRScheduler):
    """
    Fine-tuning methods from the paper
    "[arXiv:1801.06146]Universal Language Model Fine-tuning for Text Classification".

    Specifically, modifies training schedule using slanted triangular learning rates,
    discriminative fine-tuning (per-layer learning rates), and gradual unfreezing.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        cut_frac: the fraction of iterations we increase the learning rate. Default 0.1
        ratio (int): how much smaller the lowest LR is from the maximum LR eta_max.
            Default: 32.
        non_pretrained_param_groups (int): Number of param_groups, starting from the
            end, that were not pretrained. The default value is 2, since the base Model
            class supplies to the optimizer typically one param_group from the embedding
            and one param_group from its other components.
        lm_lr_multiplier (float): Factor to multiply lr for all pretrained layers by.
        lm_use_per_layer_lr (bool): Whether to make each pretrained layer's lr
            one-half as large as the next (higher) layer.
        lm_gradual_unfreezing (bool): Whether to unfreeze layers one by one (per epoch).
        last_epoch (int): Though the name is `last_epoch`, it means `last batch update`.
            last_batch_update: = current_epoch_number * num_batches_per_epoch + batch_id
            after each batch update, it will increment 1
    """

    def __init__(
        self,
        optimizer,
        cut_frac=0.1,
        ratio=32,
        non_pretrained_param_groups=2,
        lm_lr_multiplier=1.0,
        lm_use_per_layer_lr=False,
        lm_gradual_unfreezing=True,
        last_epoch=-1,
    ):
        assert isinstance(optimizer, torch.optim.Adam)
        self.num_epochs = None  # to be set later by Trainer
        self.steps_per_epoch = None  # to be set later by Trainer
        self.cut_frac = cut_frac
        self.ratio = ratio

        self.lm_pretrained_layers = (
            len(optimizer.param_groups) - non_pretrained_param_groups
        )
        assert self.lm_pretrained_layers >= 0
        assert non_pretrained_param_groups > 0

        self.lm_lr_multiplier = lm_lr_multiplier
        self.lm_use_per_layer_lr = lm_use_per_layer_lr
        self.lm_gradual_unfreezing = lm_gradual_unfreezing
        super(LmFineTuning, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.num_epochs is None or self.steps_per_epoch is None:
            return [1.0] * len(self.base_lrs)

        slanted_multiplier = self._slanted_multiplier()
        return [
            (
                slanted_multiplier
                * self._lm_layer_multiplier(i)
                * self._lm_frozen_multiplier(i)
                * base_lr
            )
            for i, base_lr in enumerate(self.base_lrs)
        ]

    def _slanted_multiplier(self):
        phase_step = self.last_epoch
        phase_total_steps = self.num_epochs * self.steps_per_epoch

        if phase_step > phase_total_steps:
            return 1.0 / self.ratio

        if self.lm_gradual_unfreezing:
            unfreeze_steps = self.lm_pretrained_layers * self.steps_per_epoch

            if self.last_epoch > unfreeze_steps:
                phase_step -= unfreeze_steps
                phase_total_steps -= unfreeze_steps
            else:
                phase_step %= self.steps_per_epoch
                phase_total_steps = self.steps_per_epoch

        cut = math.floor(self.cut_frac * phase_total_steps)
        if phase_step < cut:
            p = phase_step / cut
        else:
            p = 1.0 - (phase_step - cut) / (phase_total_steps - cut)

        return (1.0 + p * (self.ratio - 1.0)) / self.ratio

    def _lm_layer_multiplier(self, layer_index):
        multiplier = 1.0

        if layer_index < self.lm_pretrained_layers:
            multiplier *= self.lm_lr_multiplier

            if self.lm_use_per_layer_lr:
                multiplier *= 2 ** (layer_index - self.lm_pretrained_layers)

        return multiplier

    def _lm_frozen_multiplier(self, layer_index):
        return 0.0 if self._lm_frozen(layer_index) else 1.0

    def _lm_frozen(self, layer_index):
        if not self.lm_gradual_unfreezing:
            return False

        if layer_index >= self.lm_pretrained_layers:
            return False

        epoch = self.last_epoch / self.steps_per_epoch
        return epoch < self.lm_pretrained_layers - layer_index


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
        self,
        optimizer: torch.optim.Optimizer,
        scheduler_params: SchedulerParams,
        lower_is_better: bool = False,
    ) -> None:
        self.batch_based_schedulers: List[_LRScheduler] = []
        self.epoch_based_schedulers: List[_LRScheduler] = []
        self.metric_based_schedulers: List[ReduceLROnPlateau] = []

        if scheduler_params.type == SchedulerType.NONE:
            pass
        elif scheduler_params.type == SchedulerType.STEP_LR:
            self.epoch_based_schedulers = [
                StepLR(optimizer, scheduler_params.step_size, scheduler_params.gamma)
            ]
        elif scheduler_params.type == SchedulerType.EXPONENTIAL_LR:
            self.epoch_based_schedulers = [
                ExponentialLR(optimizer, scheduler_params.gamma)
            ]
        elif scheduler_params.type == SchedulerType.COSINE_ANNEALING_LR:
            self.epoch_based_schedulers = [
                CosineAnnealingLR(
                    optimizer, scheduler_params.T_max, scheduler_params.eta_min
                )
            ]
        elif scheduler_params.type == SchedulerType.REDUCE_LR_ON_PLATEAU:
            self.metric_based_schedulers = [
                ReduceLROnPlateau(
                    optimizer,
                    mode="min" if lower_is_better else "max",
                    factor=scheduler_params.gamma,
                    patience=scheduler_params.patience,
                    min_lr=scheduler_params.eta_min,
                    threshold=scheduler_params.threshold,
                    threshold_mode=(
                        "abs" if scheduler_params.threshold_is_absolute else "rel"
                    ),
                    cooldown=scheduler_params.cooldown,
                )
            ]
        elif scheduler_params.type == SchedulerType.LM_FINE_TUNING_LR:
            self.batch_based_schedulers = [
                LmFineTuning(
                    optimizer,
                    scheduler_params.cut_frac,
                    scheduler_params.ratio,
                    scheduler_params.non_pretrained_param_groups,
                    scheduler_params.lm_lr_multiplier,
                    scheduler_params.lm_use_per_layer_lr,
                    scheduler_params.lm_gradual_unfreezing,
                )
            ]
        else:
            raise ValueError("Unknown optimizer scheduler type")

    def step(self, metrics: float, epoch: Optional[int] = None) -> None:
        for epoch_based_scheduler in self.epoch_based_schedulers:
            epoch_based_scheduler.step(epoch)
        for metric_based_scheduler in self.metric_based_schedulers:
            metric_based_scheduler.step(metrics, epoch)

    def step_batch(self) -> None:
        for batch_based_scheduler in self.batch_based_schedulers:
            batch_based_scheduler.step()

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
from typing import Optional, Union

import torch
from pytext.config import ConfigBase
from pytext.config.component import Component, ComponentType, create_scheduler
from pytext.optimizer import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR as TorchCosineAnnealingLR,
    CyclicLR as TorchCyclicLR,
    ExponentialLR as TorchExponentialLR,
    ReduceLROnPlateau as TorchReduceLROnPlateau,
    StepLR as TorchStepLR,
    _LRScheduler,
)


class Scheduler(Component):
    """
    Schedulers help in adjusting the learning rate during training. Scheduler
    is a wrapper class over schedulers which can be available in torch
    library or for custom implementations. There are two kinds of lr scheduling
    that is supported by this class. Per epoch scheduling and per batch scheduling.
    In per epoch scheduling, the learning rate is adjusted at the end of each epoch
    and in per batch scheduling the learning rate is adjusted after the forward and
    backward pass through one batch during the training.

    There are two main methods that needs to be implemented by the Scheduler.
    step_epoch() is called at the end of each epoch and step_batch() is called
    at the end of each batch in the training data.

    prepare() method can be used by BatchSchedulers to initialize any attributes
    they may need.

    """

    __COMPONENT_TYPE__ = ComponentType.SCHEDULER
    __EXPANSIBLE__ = True

    class Config(ConfigBase):
        pass

    def step_batch(self, **kwargs) -> None:
        pass

    def step_epoch(self, **kwargs) -> None:
        pass

    def prepare(self, train_iter, total_epochs):
        pass


class BatchScheduler(Scheduler):
    def prepare(self, train_iter, total_epochs):
        self.num_epochs = total_epochs
        self.steps_per_epoch = getattr(train_iter, "total_num_batches", None)


class LmFineTuning(_LRScheduler, BatchScheduler):
    """
    Fine-tuning methods from the paper
    "[arXiv:1801.06146]Universal Language Model Fine-tuning for Text Classification".

    Specifically, modifies training schedule using slanted triangular learning rates,
    discriminative fine-tuning (per-layer learning rates), and gradual unfreezing.
    """

    class Config(Scheduler.Config):
        #: The fraction of iterations we increase the learning rate. Default 0.1
        cut_frac: float = 0.1
        #: How much smaller the lowest LR is from the maximum LR eta_max.
        ratio: int = 32
        #: Number of param_groups, starting from the
        #: end, that were not pretrained. The default value is 2, since the base Model
        #: class supplies to the optimizer typically one param_group from the embedding
        #: and one param_group from its other components.
        non_pretrained_param_groups: int = 2
        #: Factor to multiply lr for all pretrained layers by.
        lm_lr_multiplier: float = 1.0
        #: Whether to make each pretrained layer's lr
        #:    one-half as large as the next (higher) layer.
        lm_use_per_layer_lr: bool = False
        #: Whether to unfreeze layers one by one (per epoch).
        lm_gradual_unfreezing: bool = True
        #: Though the name is `last_epoch`, it means `last batch update`.
        #: last_batch_update: = current_epoch_number * num_batches_per_epoch + batch_id
        #: after each batch update, it will increment 1
        last_epoch: int = -1

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

    @classmethod
    def from_config(cls, config: Config, optimizer):
        return cls(
            optimizer,
            config.cut_frac,
            config.ratio,
            config.non_pretrained_param_groups,
            config.lm_lr_multiplier,
            config.lm_use_per_layer_lr,
            config.lm_gradual_unfreezing,
            config.last_epoch,
        )

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

    def step_batch(self, metrics=None, epoch=None):
        self.step(epoch)


class StepLR(TorchStepLR, Scheduler):
    """
    Wrapper around `torch.optim.lr_scheduler.StepLR`
    See the original documentation for more details.
    """

    class Config(Scheduler.Config):
        #: Period of learning rate decay.
        step_size: int = 30
        #: Multiplicative factor of learning rate decay.
        gamma: float = 0.1

    @classmethod
    def from_config(cls, config: Config, optimizer):
        return cls(optimizer, config.step_size, config.gamma)

    def step_epoch(self, metrics=None, epoch=None):
        self.step(epoch)


class ReduceLROnPlateau(TorchReduceLROnPlateau, Scheduler):
    """
    Wrapper around `torch.optim.lr_scheduler.ReduceLROnPlateau`
    See the original documentation for more details.
    """

    class Config(Scheduler.Config):
        #: This indicates the desirable direction in which we would like the
        #: training to proceed. If set to true, learning rate will be reduce
        #: when quantity being monitored stops going down
        lower_is_better: bool = True
        #: Factor by which the learning rate will be reduced. new_lr = lr * factor
        factor: float = 0.1
        #: Number of epochs with no improvement after which learning rate will
        #: be reduced
        patience: int = 5
        #: Lower bound on the learning rate of all param groups
        min_lr: float = 0
        #: Threshold for measuring the new optimum, to only focus on significant
        #: changes.
        threshold: float = 0.0001
        #: One of rel, abs.
        #: In rel mode, dynamic_threshold = best * ( 1 + threshold ) in ‘max’ mode
        #: or best * ( 1 - threshold ) in min mode.
        #: In abs mode, dynamic_threshold = best + threshold in max mode or
        #: best - threshold in min mode.
        threshold_is_absolute: bool = True
        #: Number of epochs to wait before resuming normal operation after
        #: lr has been reduced.
        cooldown: int = 0

    @classmethod
    def from_config(cls, config: Config, optimizer: Optimizer):
        return cls(
            optimizer,
            mode="min" if config.lower_is_better else "max",
            factor=config.factor,
            patience=config.patience,
            min_lr=config.min_lr,
            threshold=config.threshold,
            threshold_mode=("abs" if config.threshold_is_absolute else "rel"),
            cooldown=config.cooldown,
        )

    def step_epoch(self, metrics, epoch):
        self.step(metrics, epoch)


class CosineAnnealingLR(TorchCosineAnnealingLR, BatchScheduler):
    """
    Wrapper around `torch.optim.lr_scheduler.CosineAnnealingLR`
    See the original documentation for more details.
    """

    class Config(Scheduler.Config):
        #: Maximum number of iterations.
        t_max: int = 1000
        #: Minimum learning rate
        eta_min: float = 0

    @classmethod
    def from_config(cls, config: Config, optimizer: Optimizer):
        return cls(optimizer, config.t_max, config.eta_min)

    def step_batch(self, metrics=None, epoch=None):
        self.step(epoch)


class CyclicLR(TorchCyclicLR, BatchScheduler):
    """
    Wrapper around `torch.optim.lr_scheduler.CyclicLR`
    See the original documentation for more details
    """

    class Config(Scheduler.Config):
        base_lr: float = 0.001
        max_lr: float = 0.002
        step_size_up: int = 2000
        step_size_down: Optional[int] = None
        mode: str = "triangular"
        gamma: float = 1.0
        scale_mode: str = "cycle"
        cycle_momentum: bool = True
        base_momentum: float = 0.8
        max_momentum: float = 0.9
        last_epoch: int = -1

    @classmethod
    def from_config(cls, config: Config, optimizer: Optimizer):
        return cls(
            optimizer=optimizer,
            base_lr=config.base_lr,
            max_lr=config.max_lr,
            step_size_up=config.step_size_up,
            step_size_down=config.step_size_down,
            mode=config.mode,
            gamma=config.gamma,
            scale_mode=config.scale_mode,
            cycle_momentum=config.cycle_momentum,
            base_momentum=config.base_momentum,
            max_momentum=config.max_momentum,
            last_epoch=config.last_epoch,
        )

    def step_batch(self, metrics=None, epoch=None):
        self.step(epoch)


class ExponentialLR(TorchExponentialLR, Scheduler):
    """
    Wrapper around `torch.optim.lr_scheduler.ExponentialLR`
    See the original documentation for more details.
    """

    class Config(Scheduler.Config):
        #: Multiplicative factor of learning rate decay.
        gamma: float = 0.1

    @classmethod
    def from_config(cls, config: Config, optimizer: Optimizer):
        return cls(optimizer, config.gamma)

    def step_epoch(self, metrics=None, epoch=None):
        self.step(epoch)


class WarmupScheduler(_LRScheduler, BatchScheduler):
    """
    Scheduler to linearly increase the learning rate from 0 to its final value over
    a number of steps:

        lr = base_lr * current_step / warmup_steps

    After the warm-up phase, the scheduler has the option of decaying the learning
    rate as the inverse square root of the number of training steps taken:

        lr = base_lr * sqrt(warmup_steps) / sqrt(current_step)
    """

    class Config(BatchScheduler.Config):
        #: number of training steps over which to increase learning rate
        warmup_steps: int = 10000

        #: whether to perform inverse sqrt decay after the warmup phase
        inverse_sqrt_decay: bool = False

    @classmethod
    def from_config(cls, config: Config, optimizer: Optimizer):
        return cls(optimizer, config.warmup_steps, config.inverse_sqrt_decay)

    def __init__(self, optimizer, warmup_steps, inverse_sqrt_decay):
        assert warmup_steps > 0
        self.warmup_steps = warmup_steps
        self.current_steps = 0
        self.inverse_sqrt_decay = inverse_sqrt_decay
        self.decay_factor = warmup_steps ** 0.5
        super().__init__(optimizer)

    def prepare(self, train_iter, total_epochs):
        super().prepare(train_iter, total_epochs)
        self.step_batch()  # initialize learning rate

    def step_batch(self):
        self.current_steps += 1
        self.step()

    def get_lr(self):
        if self.current_steps >= self.warmup_steps:
            if self.inverse_sqrt_decay:
                lr_multiplier = self.decay_factor / (self.current_steps ** 0.5)
            else:
                lr_multiplier = 1.0
        else:
            lr_multiplier = self.current_steps / self.warmup_steps
        return [lr_multiplier * base_lr for base_lr in self.base_lrs]


class PolynomialDecayScheduler(_LRScheduler, BatchScheduler):
    """
    Applies a polynomial decay with lr warmup to the learning rate.

    It is commonly observed that a monotonically decreasing learning rate, whose
    degree of change is carefully chosen, results in a better performing model.

    This scheduler linearly increase learning rate from 0 to final value at the
    beginning of training, determined by warmup_steps.
    Then it applies a polynomial decay function to an optimizer step, given a
    provided `base_lrs` to reach an `end_learning_rate` after `total_steps`.
    """

    class Config(BatchScheduler.Config):
        #: number of training steps over which to increase learning rate
        warmup_steps: int = 0
        #: number of training steps for learning rate decay
        total_steps: int
        #: end learning rate after `total_steps` of training
        end_learning_rate: float
        #: power used for polynomial decay calculation
        power: float = 1.0

    @classmethod
    def from_config(cls, config: Config, optimizer: Optimizer):
        return cls(
            optimizer,
            config.warmup_steps,
            config.total_steps,
            config.end_learning_rate,
            config.power,
        )

    def __init__(self, optimizer, warmup_steps, total_steps, end_learning_rate, power):
        assert total_steps > warmup_steps >= 0
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.current_steps = 0
        super().__init__(optimizer)

    def prepare(self, train_iter, total_epochs):
        super().prepare(train_iter, total_epochs)
        self.step_batch()  # initialize learning rate

    def get_lr(self):
        if self.current_steps <= self.warmup_steps:
            # during warmup the learning rate linearly increases until
            # it reaches base_lr.
            warmup_factor = self.current_steps / self.warmup_steps
            lrs = [warmup_factor * base_lr for base_lr in self.base_lrs]
        elif self.current_steps <= self.total_steps:
            # start polynomial weight decay until it reaches end_learning_rate
            decay_factor = (
                1
                - (self.current_steps - self.warmup_steps)
                / (self.total_steps - self.warmup_steps)
            ) ** self.power

            lrs = [
                (base_lr - self.end_learning_rate) * decay_factor
                + self.end_learning_rate
                for base_lr in self.base_lrs
            ]
        else:
            # reach end_learning_rate after total_steps
            lrs = [self.end_learning_rate for _ in self.base_lrs]

        return lrs

    def step_batch(self):
        self.current_steps += 1
        # update optimizer.param_groups's learning rate
        self.step()


class SchedulerWithWarmup(_LRScheduler, BatchScheduler):
    """
    Wraps another scheduler with a warmup phase. After `warmup_steps` defined in
    warmup_scheduler.warmup_steps, the scheduler will switch to use the specified
    scheduler in `scheduler`.

    `warmup_scheduler`: is the configuration for the WarmupScheduler, that warms up
    learning rate over `warmup_steps` linearly.

    `scheduler`: is the main scheduler that will be applied after the warmup phase
    (once `warmup_steps` have passed)
    """

    class Config(BatchScheduler.Config):
        # the definition of the warmup scheduler for the warmup phase
        warmup_scheduler: WarmupScheduler.Config = WarmupScheduler.Config()

        # the definition of the main scheduler to apply once the warmup phase
        # has passed
        scheduler: Union[
            ExponentialLR.Config,
            CosineAnnealingLR.Config,
            ReduceLROnPlateau.Config,
            LmFineTuning.Config,
            CyclicLR.Config,
        ]

    @classmethod
    def from_config(cls, config: Config, optimizer: Optimizer):
        warmup_scheduler = create_scheduler(config.warmup_scheduler, optimizer)
        scheduler = create_scheduler(config.scheduler, optimizer)
        return cls(
            optimizer, warmup_scheduler, scheduler, config.warmup_scheduler.warmup_steps
        )

    def prepare(self, train_iter, total_epochs):
        super().prepare(train_iter, total_epochs)
        self.warmup_scheduler.prepare(train_iter, total_epochs)
        self.scheduler.prepare(train_iter, total_epochs)

    def __init__(self, optimizer, warmup_scheduler, scheduler, switch_steps):
        self.optimizer = optimizer
        self.warmup_scheduler = warmup_scheduler
        self.scheduler = scheduler
        self.switch_steps = switch_steps
        self.curr_steps = 0

    def step_batch(self):
        if self.curr_steps < self.switch_steps:
            self.curr_steps += 1
            return self.warmup_scheduler.step_batch()
        else:
            return self.scheduler.step_batch()

    def step_epoch(self, metrics, epoch):
        if self.curr_steps < self.switch_steps:
            return self.warmup_scheduler.step_epoch(metrics=metrics, epoch=epoch)
        else:
            return self.scheduler.step_epoch(metrics=metrics, epoch=None)

    def get_lr(self):
        if self.curr_steps < self.switch_steps:
            return self.warmup_scheduler.get_lr()
        else:
            return self.scheduler.get_lr()

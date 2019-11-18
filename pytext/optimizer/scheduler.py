#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math

import torch
from pytext.config import ConfigBase
from pytext.config.component import Component, ComponentType
from pytext.optimizer import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR as TorchCosineAnnealingLR,
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
    Scheduler to linearly increase learning rate from 0 to final value at the beginning
    of training.
    """

    class Config(BatchScheduler.Config):
        #: number of training steps over which to increase learning rate
        warmup_steps: int = 10000

    @classmethod
    def from_config(cls, config: Config, optimizer: Optimizer):
        return cls(optimizer, config.warmup_steps)

    def __init__(self, optimizer, warmup_steps):
        assert warmup_steps > 0
        self.warmup_steps = warmup_steps
        self.current_steps = 0
        super().__init__(optimizer)

    def prepare(self, train_iter, total_epochs):
        super().prepare(train_iter, total_epochs)
        self.step_batch()  # initialize learning rate

    def step_batch(self):
        self.current_steps += 1
        self.step()

    def get_lr(self):
        if self.current_steps >= self.warmup_steps:
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


class ReduceLRIfStationary(Scheduler):
    """Reduce learning rate by a factor when stationary phase of an stochastic
    optimizer is detected. Inspired by "Convergence diagnostics for stochastic
    gradient descent with constant learning rate" by Chee and Toulis (2018). This
    scheduler is very similar to ReduceLROnPlateau in spirit; however it detects
    the stationary phase of optimization using inner product between consecutive
    (batch) stochastic gradients instead of relying on additional eval set.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        factor (float): Factor by which the learning rate will be
            reduced. new_lr = lr * factor. Default: 0.1.
        burnin (int): if the average consecutive inner prod in burnin in steps are
            below zero, lr will be reduced. After lr is reduced, the step counter
            and averaged consecutive will be reset. Default: 10
        min_lr (float or list): A scalar or a list of scalars. A
            lower bound on the learning rate of all param groups
            or each group respectively. Default: 0.
        max_lr (float or list): A scalar or a list of scalars. A
            upper bound on the learning rate of all param groups
            or each group respectively. Default: 10.
        eps (float): Minimal decay applied to lr. If the difference
            between new and old lr is smaller than eps, the update is
            ignored. Default: 1e-8.

    """

    class Config(Scheduler.Config):
        factor: float = 0.8
        burnin: int = 100
        min_lr: float = 0
        max_lr: float = 10
        eps: float = 1e-8

    @classmethod
    def from_config(cls, config: Config, optimizer: Optimizer):
        return cls(
            optimizer,
            factor=config.factor,
            burnin=config.burnin,
            min_lr=config.min_lr,
            max_lr=config.max_lr,
            eps=config.eps,
        )

    def __init__(
        self, optimizer, factor=0.8, burnin=100, min_lr=0, max_lr=10, eps=1e-8
    ):
        if factor >= 1.0:
            raise ValueError("Factor should be < 1.0.")
        self.factor = factor

        self.burnin = burnin
        if not isinstance(optimizer, Optimizer):
            raise TypeError("{} is not an Optimizer".format(type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(min_lr, list) or isinstance(min_lr, tuple):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError(
                    "expected {} min_lrs, got {}".format(
                        len(optimizer.param_groups), len(min_lr)
                    )
                )
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        if isinstance(max_lr, list) or isinstance(max_lr, tuple):
            if len(max_lr) != len(optimizer.param_groups):
                raise ValueError(
                    "expected {} min_lrs, got {}".format(
                        len(optimizer.param_groups), len(max_lr)
                    )
                )
            self.max_lrs = list(max_lr)
        else:
            self.max_lrs = [max_lr] * len(optimizer.param_groups)

        self.eps = eps
        self.prev_param_group = None
        self.accumulate_consecutive_inner_product = 0.0
        self.steps_since_last_change = 0

    def _compute_grad_innerprod(self, param_grad_group1, param_grad_group2):
        assert len(param_grad_group1) == len(
            param_grad_group2
        ), "the gradient dimension between two model updates changed"
        innerprod = 0.0
        for g1, g2 in zip(param_grad_group1, param_grad_group2):
            assert g1.shape == g2.shape, "gradient dimension does not match"
            innerprod += torch.matmul(g1, g2).item()
        return innerprod

    def step_batch(self):
        param_grad_group = []
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    param_grad_group.append(torch.flatten(p.grad.clone().detach()))

        if self.prev_param_group is None:
            self.prev_param_group = param_grad_group
            return

        innerprod = self._compute_grad_innerprod(
            self.prev_param_group, param_grad_group
        )
        self.accumulate_consecutive_inner_product += innerprod
        self.prev_param_group = param_grad_group
        self.steps_since_last_change += 1
        if self.steps_since_last_change >= self.burnin:
            if self.accumulate_consecutive_inner_product <= 0:
                # stationary phase, reduce lr
                self._reduce_lr()
            else:
                self._enlarge_lr()
            # reset the history
            self.steps_since_last_change = 0
            self.accumulate_consecutive_inner_product = 0
            self.prev_param_group = None

    def _reduce_lr(self):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group["lr"])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                print(f"new lr is {new_lr}")
                param_group["lr"] = new_lr

    def _enlarge_lr(self):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group["lr"])
            new_lr = min(old_lr / self.factor, self.max_lrs[i])
            if new_lr - old_lr > self.eps:
                print(f"new lr is {new_lr}")
                param_group["lr"] = new_lr

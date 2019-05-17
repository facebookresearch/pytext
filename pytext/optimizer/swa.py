#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import warnings
from collections import defaultdict
from typing import Union

import torch
from pytext.config.component import create_optimizer
from pytext.optimizer.optimizers import SGD, Adam, Optimizer
from torch.optim import Optimizer as PT_Optimizer


class StochasticWeightAveraging(Optimizer, PT_Optimizer):
    class Config(Optimizer.Config):
        optimizer: Union[SGD.Config, Adam.Config] = SGD.Config()
        start: int = 10
        frequency: int = 5
        swa_learning_rate: float = 0.05

    def __init__(self, optimizer, swa_start=None, swa_freq=None, swa_lr=None):
        r"""Implements Stochastic Weight Averaging (SWA).

        THIS WAS DIRECTLY COPIED OVER FROM pytorch/contrib
            https://github.com/pytorch/contrib
        Stochastic Weight Averaging was proposed in `Averaging Weights Leads to
        Wider Optima and Better Generalization`_ by Pavel Izmailov, Dmitrii
        Podoprikhin, Timur Garipov, Dmitry Vetrov and Andrew Gordon Wilson
        (UAI 2018).

        SWA is implemented as a wrapper class taking optimizer instance as input
        and applying SWA on top of that optimizer.

        SWA can be used in two modes: automatic and manual. In the automatic
        mode SWA running averages are automatically updated every
        :attr:`swa_freq` steps after :attr:`swa_start` steps of optimization. If
        :attr:`swa_lr` is provided, the learning rate of the optimizer is reset
        to :attr:`swa_lr` at every step starting from :attr:`swa_start`. To use
        SWA in automatic mode provide values for both :attr:`swa_start` and
        :attr:`swa_freq` arguments.

        Alternatively, in the manual mode, use :meth:`update_swa` or
        :meth:`update_swa_group` methods to update the SWA running averages.

        In the end of training use `swap_swa_sgd` method to set the optimized
        variables to the computed averages.

        Args:
            optimizer (torch.optim.Optimizer): optimizer to use with SWA
            swa_start (int): number of steps before starting to apply SWA in
                automatic mode; if None, manual mode is selected (default: None)
            swa_freq (int): number of steps between subsequent updates of
                SWA running averages in automatic mode; if None, manual mode is
                selected (default: None)
            swa_lr (float): learning rate to use starting from step swa_start
                in automatic mode; if None, learning rate is not changed
                (default: None)

        .. _Averaging Weights Leads to Wider Optima and Better Generalization:
            https://arxiv.org/abs/1803.05407
        .. _Improving Consistency-Based Semi-Supervised Learning with Weight
            Averaging:
            https://arxiv.org/abs/1806.05594
        """
        self._auto_mode, (self.swa_start, self.swa_freq) = self._check_params(
            self, swa_start, swa_freq
        )
        self.swa_lr = swa_lr

        if self._auto_mode:
            if swa_start < 0:
                raise ValueError("Invalid swa_start: {}".format(swa_start))
            if swa_freq < 1:
                raise ValueError("Invalid swa_freq: {}".format(swa_freq))
        else:
            if self.swa_lr is not None:
                warnings.warn("Some of swa_start, swa_freq is None, ignoring swa_lr")
            # If not in auto mode make all swa parameters None
            self.swa_lr = None
            self.swa_start = None
            self.swa_freq = None

        if self.swa_lr is not None and self.swa_lr < 0:
            raise ValueError("Invalid SWA learning rate: {}".format(swa_lr))

        self.optimizer = optimizer

        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.opt_state = self.optimizer.state
        for group in self.param_groups:
            group["n_avg"] = 0
            group["step_counter"] = 0

    @staticmethod
    def _check_params(self, swa_start, swa_freq):
        params = [swa_start, swa_freq]
        params_none = [param is None for param in params]
        if not all(params_none) and any(params_none):
            warnings.warn("Some of swa_start, swa_freq is None, ignoring other")
        for i, param in enumerate(params):
            if param is not None and not isinstance(param, int):
                params[i] = int(param)
                warnings.warn("Casting swa_start, swa_freq to int")
        return not any(params_none), params

    def _reset_lr_to_swa(self):
        if self.swa_lr is None:
            return
        for param_group in self.param_groups:
            if param_group["step_counter"] >= self.swa_start:
                param_group["lr"] = self.swa_lr

    def update_swa_group(self, group):
        r"""Updates the SWA running averages for the given parameter group.

        Arguments:
            param_group (dict): Specifies for what parameter group SWA running
                averages should be updated

        Examples:
            >>> # automatic mode
            >>> base_opt = torch.optim.SGD([{'params': [x]},
            >>>             {'params': [y], 'lr': 1e-3}], lr=1e-2, momentum=0.9)
            >>> opt = torchcontrib.optim.SWA(base_opt)
            >>> for i in range(100):
            >>>     opt.zero_grad()
            >>>     loss_fn(model(input), target).backward()
            >>>     opt.step()
            >>>     if i > 10 and i % 5 == 0:
            >>>         # Update SWA for the second parameter group
            >>>         opt.update_swa_group(opt.param_groups[1])
            >>> opt.swap_swa_sgd()
        """
        for p in group["params"]:
            param_state = self.state[p]
            if "swa_buffer" not in param_state:
                param_state["swa_buffer"] = torch.zeros_like(p.data)
            buf = param_state["swa_buffer"]
            virtual_decay = 1 / float(group["n_avg"] + 1)
            diff = (p.data - buf) * virtual_decay
            buf.add_(diff)
        group["n_avg"] += 1

    def update_swa(self):
        r"""Updates the SWA running averages of all optimized parameters.
        """
        for group in self.param_groups:
            self.update_swa_group(group)

    def finalize(self):
        r"""Swaps the values of the optimized variables and swa buffers.

        It's meant to be called in the end of training to use the collected
        swa running averages. It can also be used to evaluate the running
        averages during training; to continue training `swap_swa_sgd`
        should be called again.
        """
        for group in self.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                if "swa_buffer" not in param_state:
                    # If swa wasn't applied we don't swap params
                    warnings.warn(
                        "SWA wasn't applied to param {}; skipping it".format(p)
                    )
                    continue
                buf = param_state["swa_buffer"]
                tmp = torch.empty_like(p.data)
                tmp.copy_(p.data)
                p.data.copy_(buf)
                buf.copy_(tmp)
        return True

    def step(self, closure=None):
        r"""Performs a single optimization step.

        In automatic mode also updates SWA running averages.
        """
        self._reset_lr_to_swa()
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            group["step_counter"] += 1
            steps = group["step_counter"]
            if self._auto_mode:
                if steps > self.swa_start and steps % self.swa_freq == 0:
                    self.update_swa_group(group)
        return loss

    def state_dict(self):
        r"""Returns the state of SWA as a :class:`dict`.

        It contains three entries:
            * opt_state - a dict holding current optimization state of the base
                optimizer. Its content differs between optimizer classes.
            * swa_state - a dict containing current state of SWA. For each
                optimized variable it contains swa_buffer keeping the running
                average of the variable
            * param_groups - a dict containing all parameter groups
        """
        opt_state_dict = self.optimizer.state_dict()
        swa_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        opt_state = opt_state_dict["state"]
        param_groups = opt_state_dict["param_groups"]
        return {
            "opt_state": opt_state,
            "swa_state": swa_state,
            "param_groups": param_groups,
        }

    def load_state_dict(self, state_dict):
        r"""Loads the optimizer state.

        Args:
            state_dict (dict): SWA optimizer state. Should be an object returned
                from a call to `state_dict`.
        """
        swa_state_dict = {
            "state": state_dict["swa_state"],
            "param_groups": state_dict["param_groups"],
        }
        opt_state_dict = {
            "state": state_dict["opt_state"],
            "param_groups": state_dict["param_groups"],
        }
        PT_Optimizer.load_state_dict(self, swa_state_dict)
        self.optimizer.load_state_dict(opt_state_dict)
        self.opt_state = self.optimizer.state

    def add_param_group(self, param_group):
        r"""Add a param group to the :class:`Optimizer` s `param_groups`.

        This can be useful when fine tuning a pre-trained network as frozen
        layers can be made trainable and added to the :class:`Optimizer` as
        training progresses.

        Args:
            param_group (dict): Specifies what Tensors should be optimized along
            with group specific optimization options.
        """
        param_group["n_avg"] = 0
        param_group["step_counter"] = 0
        self.optimizer.add_param_group(param_group)

    @staticmethod
    def bn_update(loader, model, device=None):
        r"""Updates BatchNorm running_mean, running_var buffers in the model.

        It performs one pass over data in `loader` to estimate the activation
        statistics for BatchNorm layers in the model.

        Args:
            loader (torch.utils.data.DataLoader): dataset loader to compute the
                activation statistics on. Each data batch should be either a
                tensor, or a list/tuple whose first element is a tensor
                containing data.

            model (torch.nn.Module): model for which we seek to update BatchNorm
                statistics.

            device (torch.device, optional): If set, data will be trasferred to
                :attr:`device` before being passed into :attr:`model`.
        """
        if not _check_bn(model):
            return
        was_training = model.training
        model.train()
        momenta = {}
        model.apply(_reset_bn)
        model.apply(lambda module: _get_momenta(module, momenta))
        n = 0
        for input in loader:
            if isinstance(input, (list, tuple)):
                input = input[0]
            b = input.size(0)

            momentum = b / float(n + b)
            for module in momenta.keys():
                module.momentum = momentum

            if device is not None:
                input = input.to(device)

            model(input)
            n += b

        model.apply(lambda module: _set_momenta(module, momenta))
        model.train(was_training)

    @classmethod
    def from_config(cls, config: Config, model: torch.nn.Module):
        base_opt = create_optimizer(config.optimizer, model)
        return cls(base_opt, config.start, config.frequency, config.swa_learning_rate)


# BatchNorm utils
def _check_bn_apply(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def _check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn_apply(module, flag))
    return flag[0]


def _reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]

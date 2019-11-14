#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from pytext.config import ConfigBase
from pytext.config.component import Component, ComponentType


class Optimizer(Component):
    __COMPONENT_TYPE__ = ComponentType.OPTIMIZER
    __EXPANSIBLE__ = True

    class Config(ConfigBase):
        pass

    def backward(self, loss):
        loss.backward()

    def clip_grad_norm(self, max_norm, model=None):
        if max_norm is None:
            """incase max_norm is none we don't compute clip_grad_norm.
            """
            return None
        elif model is None:
            """Some callers are passing max_norm only instead of both the args.
               For those we treat model as max_norm.
               eg. optimizer.clip_grad_norm(max_norm)
            """
            return torch.nn.utils.clip_grad_norm_(self.params, max_norm)
        else:
            return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

    def pre_export(self, model):
        pass

    def finalize(self) -> bool:
        return False

    def multiply_grads(self, c):
        """Multiplies grads by a constant *c*."""
        for p in self.params:
            if p.grad is not None:
                p.grad.data.mul_(c)

    @property
    def params(self):
        """Return an iterable of the parameters held by the optimizer."""
        for param_group in self.param_groups:
            for p in param_group["params"]:
                yield p


class Adagrad(torch.optim.Adagrad, Optimizer):
    class Config(Optimizer.Config):
        lr: float = 1e-2
        weight_decay: float = 0.00001

    def __init__(self, parameters, lr, weight_decay):
        super().__init__(parameters, lr=lr, weight_decay=weight_decay)

    @classmethod
    def from_config(cls, config: Config, model: torch.nn.Module):
        return cls(model.parameters(), config.lr, config.weight_decay)


class Adam(torch.optim.Adam, Optimizer):
    class Config(Optimizer.Config):
        lr: float = 0.001
        weight_decay: float = 0.00001
        eps: float = 1e-8

    def __init__(self, parameters, lr, weight_decay, eps):
        super().__init__(parameters, lr=lr, weight_decay=weight_decay, eps=eps)

    @classmethod
    def from_config(cls, config: Config, model: torch.nn.Module):
        return cls(model.parameters(), config.lr, config.weight_decay, config.eps)


class SGD(torch.optim.SGD, Optimizer):
    class Config(Optimizer.Config):
        lr: float = 0.001
        momentum: float = 0.0

    def __init__(self, parameters, lr, momentum):
        super().__init__(parameters, lr=lr, momentum=momentum)

    @classmethod
    def from_config(cls, config: Config, model: torch.nn.Module):
        return cls(model.parameters(), config.lr, config.momentum)


class AdamW(torch.optim.AdamW, Optimizer):
    """Adds PyText support for
       Decoupled Weight Decay Regularization for Adam as done in the paper:
       https://arxiv.org/abs/1711.05101
       for more information read the fast.ai blog on this optimization
       method here: https://www.fast.ai/2018/07/02/adam-weight-decay/
    """

    class Config(Optimizer.Config):
        lr: float = 0.001
        weight_decay: float = 1e-2
        eps: float = 1e-8

    def __init__(self, parameters, lr, weight_decay, eps):
        super().__init__(parameters, lr=lr, weight_decay=weight_decay, eps=eps)

    @classmethod
    def from_config(cls, config: Config, model: torch.nn.Module):
        return cls(model.parameters(), config.lr, config.weight_decay, config.eps)


def learning_rates(optimizer):
    for param_group in optimizer.param_groups:
        yield param_group["lr"]

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

    def finalize(self) -> bool:
        return False


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


def learning_rates(optimizer):
    for param_group in optimizer.param_groups:
        yield param_group["lr"]

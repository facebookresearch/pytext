#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List

import torch
from pytext.config.pytext_config import OptimizerParams, OptimizerType
from pytext.models.model import Model


def create_optimizer(
    model: Model, optimizer_params: OptimizerParams
) -> List[torch.optim.Optimizer]:
    if optimizer_params.type == OptimizerType.ADAM:
        return [
            torch.optim.Adam(
                model.get_param_groups_for_optimizer(),
                lr=optimizer_params.lr,
                weight_decay=optimizer_params.weight_decay,
            )
        ]
    elif optimizer_params.type == OptimizerType.SGD:
        return [
            torch.optim.SGD(
                model.get_param_groups_for_optimizer(),
                lr=optimizer_params.lr,
                momentum=optimizer_params.momentum,
            )
        ]
    else:
        raise ValueError("Unknown optimizer type")


def optimizer_zero_grad(optimizers: List[torch.optim.Optimizer]) -> None:
    for op in optimizers:
        op.zero_grad()


def optimizer_step(optimizers: List[torch.optim.Optimizer]) -> None:
    for op in optimizers:
        op.step()


def learning_rates(optimizers):
    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            yield param_group["lr"]

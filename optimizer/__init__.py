#!/usr/bin/env python3

from typing import Dict, Iterator, List, Union, ValuesView

import torch
from pytext.config.pytext_config import (
    OptimizerParams,
    OptimizerType,
    SchedulerParams,
    SchedulerType,
)
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ExponentialLR,
    StepLR,
    _LRScheduler,
)


def create_optimizer(
    model: torch.nn.Module, optimizer_params: OptimizerParams
) -> List[torch.optim.Optimizer]:
    if optimizer_params.type == OptimizerType.ADAM:
        sparse_grads_params: Dict[str, torch.nn.Parameter] = {}
        if hasattr(model, "get_model_params_for_optimizer"):
            sparse_grads_params, dense_grads_params = (
                model.get_model_params_for_optimizer()
            )
        if sparse_grads_params:
            print(
                "Using sparse gradients for the following parameters: {}.".format(
                    list(sparse_grads_params.keys())
                )
            )
            print(
                "Using dense gradients for the following parameters: {}.".format(
                    list(dense_grads_params.keys())
                )
            )
            return [
                torch.optim.SparseAdam(
                    sparse_grads_params.values(), lr=optimizer_params.lr
                ),
                torch.optim.Adam(
                    dense_grads_params.values(),
                    lr=optimizer_params.lr,
                    weight_decay=optimizer_params.weight_decay,
                ),
            ]
        else:
            return [
                torch.optim.Adam(
                    get_params(model.parameters()),
                    lr=optimizer_params.lr,
                    weight_decay=optimizer_params.weight_decay,
                )
            ]
    elif optimizer_params.type == OptimizerType.SGD:
        return [
            torch.optim.SGD(
                get_params(model.parameters()),
                lr=optimizer_params.lr,
                momentum=optimizer_params.momentum,
            )
        ]
    else:
        raise ValueError("Unknown optimizer type")


def create_scheduler(
    optimizers: List[torch.optim.Optimizer], scheduler_params: SchedulerParams
) -> List[_LRScheduler]:

    if scheduler_params.type == SchedulerType.NONE:
        return []

    if scheduler_params.type == SchedulerType.STEP_LR:
        return [
            StepLR(optimizer, scheduler_params.step_size, scheduler_params.gamma)
            for optimizer in optimizers
        ]

    if scheduler_params.type == SchedulerType.EXPONENTIAL_LR:
        return [
            ExponentialLR(optimizer, scheduler_params.gamma) for optimizer in optimizers
        ]

    if scheduler_params.type == SchedulerType.COSINE_ANNEALING_LR:
        return [
            CosineAnnealingLR(
                optimizer, scheduler_params.T_max, scheduler_params.eta_min
            )
            for optimizer in optimizers
        ]

    raise ValueError("Unknown optimizer scheduler type")


def get_params(
    params: Union[Iterator[torch.nn.Parameter], Dict[str, torch.Tensor]]
) -> Union[ValuesView[torch.nn.Parameter], Iterator[torch.nn.Parameter]]:
    """
    The C++ autogradpp implementation returns a map<string, Variable> which
    gets translated into a dict by PyBind. See https://fburl.com/h3z961jd.
    We need  to handle this because in RNNG we use autogradpp's implementation.
    """
    if isinstance(params, dict):
        return params.values()
    return params


def optimizer_zero_grad(optimizers: List[torch.optim.Optimizer]) -> None:
    for op in optimizers:
        op.zero_grad()


def optimizer_step(optimizers: List[torch.optim.Optimizer]) -> None:
    for op in optimizers:
        op.step()


def scheduler_step(schedulers: List[torch.optim.lr_scheduler._LRScheduler]) -> None:
    for scheduler in schedulers:
        scheduler.step()

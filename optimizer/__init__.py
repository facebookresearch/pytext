#!/usr/bin/env python3

from typing import Dict, Iterator, List, Tuple, Union, ValuesView

import torch
from pytext.config.pytext_config import OptimizerParams, OptimizerType

# TODO remove it after migrating nnlg to new config
from pytext.config.ttypes import OptimizerType as OptimizerTypeThrift


def create_optimizer(
    model: torch.nn.Module, optimizer_params: OptimizerParams
) -> List[torch.optim.Optimizer]:
    if optimizer_params.type in [OptimizerType.ADAM, OptimizerTypeThrift.ADAM]:
        # TODO hack workaround, should remove optimizer_params.type == OptimizerType.ADAM
        # after migrating nnlg config
        if (
            optimizer_params.type == OptimizerType.ADAM
            and hasattr(model, "embedding") and model.embedding.word_embed.sparse
        ):
            embeddings_params, other_params = split_model_params(model)
            return [
                torch.optim.SparseAdam(embeddings_params, lr=optimizer_params.lr),
                torch.optim.Adam(
                    other_params,
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
    elif optimizer_params.type in [OptimizerType.SGD, OptimizerTypeThrift.SGD]:
        return [
            torch.optim.SGD(
                get_params(model.parameters()),
                lr=optimizer_params.lr,
                momentum=optimizer_params.momentum,
            )
        ]
    else:
        raise ValueError("Unknown optimizer type")


def split_model_params(
    model: torch.nn.Module
) -> Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    """
    Current implementation makes a distinction between embedding params append
    rest of the model params because we're only supporting sparse gradients
    for embeddings. This is not generic enough that allows the flexibility of
    using sparse optimizer for any subset of parameters in the model.
    """
    embedding_params_name = {
        n for n, _ in model.embedding.word_embed.named_parameters()
    }
    other_params = []
    for name, param in model.named_parameters():
        if name not in embedding_params_name:
            other_params.append(param)
    return model.embedding.word_embed.parameters(), other_params


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

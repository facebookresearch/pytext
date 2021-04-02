#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from enum import Enum
from typing import Union

import torch
import torch.nn.functional as F
from pytext.config import ConfigBase
from pytext.config.component import create_loss

from .loss import HingeLoss, Loss, NLLLoss


class CostFunctionType(Enum):
    HAMMING = "hamming"


def hamming_distance(logits, targets, cost_scale=1.0):
    """
    Computes Hamming distance (https://en.wikipedia.org/wiki/Hamming_distance), which is
    defined as the number of positions where two sequences of equal length differ. We apply
    Hamming distance locally, incrementing non-gold token scores by `cost_scale`.

    ```
    Example:
    Given targets = [0, 1] and cost_scale = 1.0, we have the following:
    logits (before) = [[-1.0, 1.0, 2.0], [-2.0, -1.0, 1.0]]
    logits (after) = [[-1.0, 2.0, 3.0], [-1.0, -1.0, 2.0]]
    ```
    """

    hamming_cost = cost_scale * torch.ones_like(logits)  # B x T x V
    gold_cost = torch.zeros_like(targets).to(logits.dtype).unsqueeze(2)  # B x T x 1
    hamming_cost.scatter_(2, targets.unsqueeze(2), gold_cost)

    return hamming_cost


def get_cost_fn(cost_fn_type: CostFunctionType):
    """Retrieves a cost function corresponding to `cost_fn_type`."""

    if cost_fn_type == cost_fn_type.HAMMING:
        return hamming_distance
    else:
        raise RuntimeError("invalid cost type provideo")


class StructuredLoss(Loss):
    """Generic loss function applied to structured outputs."""

    def __init__(self, config, ignore_index=1):
        self.ignore_index = ignore_index

    def __call__(self, logits, targets, reduce=True):
        raise NotImplementedError


class StructuredMarginLoss(StructuredLoss):
    """
    Margin-based loss which requires a gold structure Y to score at least
    `cost(Y, Y')` above a hypothesis structure `Y'`. The cost function used is
    variable, but should reflect the underlying semantics of the task (e.g.,
    BLEU in machine translation).
    """

    class Config(ConfigBase):
        cost_scale: float = 1.0
        cost_fn: CostFunctionType = CostFunctionType.HAMMING
        label_loss: Union[NLLLoss.Config, HingeLoss.Config] = NLLLoss.Config()

    def __init__(self, config, ignore_index=1, *args, **kwargs):
        super().__init__(config, ignore_index)

        self.cost_scale = config.cost_scale
        self.cost_fn = get_cost_fn(config.cost_fn)

        self.label_loss_fn = create_loss(config.label_loss, ignore_index=ignore_index)

    def __call__(self, logits, targets, reduce=True):
        # Get cost-augmented logits.
        cost = self.cost_fn(logits, targets, self.cost_scale)
        logits = logits.clone() + cost

        # NLLLoss expects log normalized logits.
        if isinstance(self.label_loss_fn, NLLLoss):
            logits = F.log_softmax(logits, 2)

        # Flatten logits and targets.
        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1)

        return self.label_loss_fn(logits, targets, reduce)

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import List

import torch
import torch.nn as nn
from pytext.common.constants import BatchContext
from pytext.models.model import Model


class DisjointMultitaskModel(Model):
    """
    Wrapper model to train multiple PyText models that share parameters.
    Designed to be used for multi-tasking when the tasks have disjoint datasets.

    Modules which have the same shared_module_key and type share parameters.
    Only need to configure the first such module in full in each case.

    Args:
        models (type): Dictionary of models of sub-tasks.

    Attributes:
        current_model (type): Current model to route the input batch to.

    """

    def __init__(self, models) -> None:
        models = nn.ModuleDict(models)
        super().__init__(None, None, None, None)
        self.models = models
        self.current_model = next(iter(models.values()))

    def contextualize(self, context):
        self.current_model = self.models[context[BatchContext.TASK_NAME]]

    def get_loss(self, logits, targets, context):
        return self.current_model.get_loss(logits, targets, context)

    def get_pred(self, logits, targets, context, *args):
        return self.current_model.get_pred(logits, targets, context, *args)

    def forward(self, *inputs) -> List[torch.Tensor]:
        return self.current_model.forward(*inputs)

    def state_dict(self):
        # This is called during pickle, we don't want the current_model copied
        model, self.current_model = self.current_model, None
        try:
            return super().state_dict()
        finally:
            self.current_model = model

    def load_state_dict(self, state_dict, strict=True):
        self.current_model = None
        super().load_state_dict(state_dict, strict)

    def save_modules(self, base_path, suffix=""):
        for name, model in self.models.items():
            model.save_modules(base_path, f"-{name}{suffix}")

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import List

import torch
import torch.nn as nn
from pytext.common.constants import BatchContext
from pytext.models.model import Model
from pytext.utils.usage import log_class_usage


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

    def __init__(self, models, loss_weights) -> None:
        models = nn.ModuleDict(models)
        super().__init__(None, None, None, None)
        self.models = models
        # make this a list to prevent registering in state_dict
        self._current_model = [next(iter(models.values()))]
        self.loss_weights = loss_weights
        log_class_usage(__class__)

    def contextualize(self, context):
        self._current_model[0] = self.models[context[BatchContext.TASK_NAME]]
        self.current_loss_weight = self.loss_weights[context[BatchContext.TASK_NAME]]

    @property
    def current_model(self):
        return self._current_model[0]

    def get_loss(self, logits, targets, context):
        return self.current_loss_weight * self.current_model.get_loss(
            logits, targets, context
        )

    def get_pred(self, logits, targets=None, context=None, *args):
        return self.current_model.get_pred(logits, targets, context, *args)

    def forward(self, *inputs) -> List[torch.Tensor]:
        return self.current_model.forward(*inputs)

    def save_modules(self, base_path, suffix=""):
        for name, model in self.models.items():
            model.save_modules(base_path, f"-{name}{suffix}")


class NewDisjointMultitaskModel(DisjointMultitaskModel):
    def arrange_model_inputs(self, tensor_dict):
        self.contextualize(tensor_dict)
        return self.current_model.arrange_model_inputs(tensor_dict)

    def arrange_targets(self, tensor_dict):
        return self.current_model.arrange_targets(tensor_dict)

    def arrange_model_context(self, tensor_dict):
        return self.current_model.arrange_model_context(tensor_dict)

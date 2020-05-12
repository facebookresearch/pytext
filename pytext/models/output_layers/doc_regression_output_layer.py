#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Any, Dict, List, Optional

import torch
from pytext.config.component import create_loss
from pytext.loss import MSELoss
from pytext.utils.usage import log_class_usage

from .output_layer_base import OutputLayerBase


class RegressionOutputLayer(OutputLayerBase):
    """
    Output layer for doc regression models. Currently only supports Mean Squared Error
    loss.

    Args:
        loss (MSELoss): config for MSE loss
        squash_to_unit_range (bool): whether to clamp the output to the range [0, 1],
            via a sigmoid.
    """

    class Config(OutputLayerBase.Config):
        loss: MSELoss.Config = MSELoss.Config()
        squash_to_unit_range: bool = False

    @classmethod
    def from_config(cls, config: Config):
        return cls(create_loss(config.loss), config.squash_to_unit_range)

    def __init__(self, loss_fn: MSELoss, squash_to_unit_range: bool = False) -> None:
        super().__init__()
        self.loss_fn = loss_fn
        self.squash_to_unit_range = squash_to_unit_range
        log_class_usage(__class__)

    def get_loss(
        self,
        logit: torch.Tensor,
        target: torch.Tensor,
        context: Optional[Dict[str, Any]] = None,
        reduce: bool = True,
    ) -> torch.Tensor:
        """
        Compute regression loss from logits and targets.

        Args:
            logit (torch.Tensor): Logits returned :class:`~pytext.models.Model`.
            target (torch.Tensor): True label/target to compute loss against.
            context (Optional[Dict[str, Any]]): Context is a dictionary of items
                that's passed as additional metadata by the
                :class:`~pytext.data.DataHandler`. Defaults to None.
            reduce (bool): Whether to reduce loss over the batch. Defaults to True.

        Returns:
            torch.Tensor: Model loss.
        """
        logit, _ = self.get_pred(logit)
        return self.loss_fn(logit, target, reduce)

    def get_pred(self, logit, *args, **kwargs):
        """
        Compute predictions and scores from the model (unlike in classification, where
        prediction = "most likely class" and scores = "log probs", here these are the
        same values). If `squash_to_unit_range` is True, fit prediction to [0, 1] via
        a sigmoid.

        Args:
            logit (torch.Tensor): Logits returned from the model.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Model prediction and scores.
        """
        prediction = logit.squeeze(dim=1)
        if self.squash_to_unit_range:
            prediction = torch.sigmoid(prediction)
        return prediction, prediction

    def torchscript_predictions(self):
        return RegressionScores(self.squash_to_unit_range)


class RegressionScores(torch.jit.ScriptModule):
    def __init__(self, squash_to_unit_range: bool):
        super().__init__()
        self.squash_to_unit_range = torch.jit.Attribute(squash_to_unit_range, bool)

    @torch.jit.script_method
    def forward(self, logits: torch.Tensor) -> List[float]:
        # logits: B x 1, prediction: B
        prediction = logits.squeeze(dim=1)
        if self.squash_to_unit_range:
            prediction = torch.sigmoid(prediction)
        scores: List[float] = prediction.tolist()
        return scores

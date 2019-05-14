#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
from typing import Any, Dict, List, Tuple

import torch
from pytext.config.component import create_loss
from pytext.data.utils import PAD
from pytext.loss import CrossEntropyLoss, Loss

from .doc_classification_output_layer import ClassificationOutputLayer
from .lm_output_layer import LMOutputLayer
from .output_layer_base import OutputLayerBase


class LMClassificationOutputLayer(LMOutputLayer):
    class Config(OutputLayerBase.Config):
        loss: CrossEntropyLoss.Config = CrossEntropyLoss.Config()

    @classmethod
    def from_config(cls, config: Config, vocab=None, labels=None):
        pad_token_idx = vocab.idx[PAD]
        return cls(
            list(vocab),
            list(labels),
            create_loss(config.loss, ignore_index=pad_token_idx),
            create_loss(config.loss),
            pad_token_idx=pad_token_idx,
        )

    def __init__(
        self,
        vocab: List[str],
        labels: List[str],
        lm_loss_fn: Loss = None,
        classification_loss_fn: Loss = None,
        config=None,
        pad_token_idx=-100,
    ):
        super().__init__(vocab, lm_loss_fn, config, pad_token_idx)
        self.classification_output = ClassificationOutputLayer(
            labels, classification_loss_fn, config
        )

    def get_loss(
        self,
        logit: torch.Tensor,
        target: torch.Tensor,
        context: Dict[str, Any],
        reduce=True,
    ) -> torch.Tensor:
        lm_logit, classification_logit = logit
        lm_target = target[:-1]
        classification_target = target[-1]
        lm_loss = super().get_loss(lm_logit, lm_target, context, reduce)
        classification_loss = self.classification_output.get_loss(
            classification_logit, classification_target, context, reduce
        )
        return lm_loss + classification_loss, lm_loss, classification_loss

    def get_pred(
        self, logit: torch.Tensor, *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return super().get_pred(logit[0], *args, **kwargs)

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytext.loss import BinaryCrossEntropyLoss, CrossEntropyLoss, Loss


class ClassificationHead(nn.Module):
    def __init__(
        self,
        is_binary: bool = True,
        label_weights: Optional[Dict[str, float]] = None,
        loss=None,
    ):
        super().__init__()
        if is_binary:
            self.loss = loss or BinaryCrossEntropyLoss(BinaryCrossEntropyLoss.Config())
        else:
            self.loss = loss or CrossEntropyLoss(CrossEntropyLoss.Config())

    def forward(self, logits) -> Tuple[torch.Tensor, torch.Tensor]:
        preds = torch.max(logits, -1)[1]
        scores = F.log_softmax(logits)
        return preds, scores

    def get_loss(self, logits, targets, reduce: bool = True):
        return self.loss(logits, targets, reduce=reduce)

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytext.loss import BinaryCrossEntropyLoss


class BinaryClassificationHead(nn.Module):
    def __init__(self, label_weights: Optional[Dict[str, float]] = None, loss=None):
        super().__init__()
        self.loss = loss or BinaryCrossEntropyLoss(BinaryCrossEntropyLoss.Config())

    def forward(self, logits):
        preds = torch.max(logits, -1)[1]
        scores = F.logsigmoid(logits)
        return preds, scores

    def get_loss(self, logits, targets, reduce: bool = True):
        return self.loss(logits, targets, reduce=reduce)

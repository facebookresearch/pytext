#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import List, Optional

import torch
import torch.nn.functional as F
from pytext.config.component import create_loss
from pytext.data.utils import Vocabulary
from pytext.fields import FieldMeta
from pytext.loss import CosineEmbeddingLoss
from pytext.models.output_layers.output_layer_base import OutputLayerBase


class PairwiseCosineDistanceOutputLayer(OutputLayerBase):
    class Config(OutputLayerBase.Config):
        loss: CosineEmbeddingLoss.Config = CosineEmbeddingLoss.Config()
        cosine_distance_threshold: float = 0.9

    @classmethod
    def from_config(
        cls,
        config,
        metadata: Optional[FieldMeta] = None,
        labels: Optional[Vocabulary] = None,
    ):
        return cls(
            list(labels), create_loss(config.loss), config.cosine_distance_threshold
        )

    def __init__(
        self,
        target_names: Optional[List[str]] = None,
        loss_fn: CosineEmbeddingLoss = None,
        cosine_distance_threshold=Config.cosine_distance_threshold,
    ):
        super().__init__(target_names, loss_fn)
        self.cosine_distance_threshold = cosine_distance_threshold

    def get_pred(self, logits: torch.Tensor, targets: torch.Tensor, *args, **kwargs):
        distances = F.cosine_similarity(logits[0], logits[1], dim=1)
        preds = (distances >= self.cosine_distance_threshold).to(dtype=torch.long)

        # Since metric reporting depends on returning a distribution over labels,
        # we will fake a distribution over two labels. We will insert the distance
        # at pred_index (pred_index = 0 or 1) in the scores tensor.
        scores = torch.zeros(logits[0].size(0), 2, device=logits[0].device)
        scores = scores.scatter(
            dim=1, index=preds.unsqueeze(1), src=distances.unsqueeze(1)
        )

        return preds, scores

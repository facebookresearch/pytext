#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from enum import IntEnum, unique
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from pytext.config.component import create_loss
from pytext.data.utils import Vocabulary
from pytext.fields import FieldMeta
from pytext.loss import (
    BinaryCrossEntropyLoss,
    CosineEmbeddingLoss,
    MAELoss,
    MSELoss,
    NLLLoss,
)
from pytext.models.output_layers.output_layer_base import OutputLayerBase
from pytext.utils.label import get_label_weights


@unique
class OutputScore(IntEnum):
    raw_cosine = 1
    norm_cosine = 2
    sigmoid_cosine = 3


class PairwiseCosineDistanceOutputLayer(OutputLayerBase):
    class Config(OutputLayerBase.Config):
        loss: Union[
            BinaryCrossEntropyLoss.Config,
            CosineEmbeddingLoss.Config,
            MAELoss.Config,
            MSELoss.Config,
            NLLLoss.Config,
        ] = CosineEmbeddingLoss.Config()
        score_threshold: float = 0.9
        score_type: OutputScore = OutputScore.norm_cosine
        label_weights: Optional[Dict[str, float]] = None

    @classmethod
    def from_config(
        cls,
        config,
        metadata: Optional[FieldMeta] = None,
        labels: Optional[Vocabulary] = None,
    ):
        label_weights = (
            get_label_weights(labels.idx, config.label_weights)
            if config.label_weights
            else None
        )
        assert (
            config.score_type == OutputScore.raw_cosine
            or config.score_type == OutputScore.norm_cosine
            or config.score_type == OutputScore.sigmoid_cosine
        ), f"Invalid score_type {config.score_type}. See OutputScore enum."
        return cls(
            list(labels),
            create_loss(config.loss, weight=label_weights),
            config.score_threshold,
            config.score_type,
        )

    def __init__(
        self,
        target_names: Optional[List[str]] = None,
        loss_fn: Union[
            BinaryCrossEntropyLoss, CosineEmbeddingLoss, MAELoss, MSELoss, NLLLoss
        ] = None,
        score_threshold: bool = Config.score_threshold,
        score_type: OutputScore = Config.score_type,
    ):
        super().__init__(target_names, loss_fn)
        self.score_threshold = score_threshold
        self.score_type = score_type

    def get_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        context: Optional[Dict[str, Any]] = None,
        reduce: bool = True,
    ) -> torch.Tensor:
        def _transform_logits(logits):
            if isinstance(self.loss_fn, CosineEmbeddingLoss):
                return logits
            elif isinstance(self.loss_fn, NLLLoss):
                cosine_sim_scores = F.cosine_similarity(logits[0], logits[1], dim=1)
                pos_scores, neg_scores = get_sigmoid_scores(cosine_sim_scores)
                return torch.log(
                    torch.cat((pos_scores.unsqueeze(1), neg_scores.unsqueeze(1)), dim=1)
                )
            else:
                return F.cosine_similarity(logits[0], logits[1], dim=1)

        def _transform_targets(targets):
            # Replace label = 0 with -1 because we're using cosine_embedding_loss.
            return (
                targets
                if isinstance(self.loss_fn, (BinaryCrossEntropyLoss, NLLLoss))
                else targets.masked_fill(mask=(targets == 0), value=-1.0).to(
                    dtype=torch.float
                )
            )

        logits = _transform_logits(logits)
        targets = _transform_targets(targets)
        return super().get_loss(logits, targets, context, reduce)

    def get_pred(self, logits: torch.Tensor, targets: torch.Tensor, *args, **kwargs):
        cosine_sim_scores = F.cosine_similarity(logits[0], logits[1], dim=1)
        if self.score_type == OutputScore.raw_cosine:
            preds = (cosine_sim_scores >= self.score_threshold).to(dtype=torch.long)

            # Since metric reporting depends on returning a distribution over labels,
            # we will fake a distribution over two labels. We will insert the distance
            # at pred_index (pred_index = 0 or 1) in the scores tensor.
            scores = torch.zeros(logits[0].size(0), 2, device=logits[0].device)
            scores = scores.scatter_(
                dim=1, index=preds.unsqueeze(1), src=cosine_sim_scores.unsqueeze(1)
            )
        else:
            pos_scores, neg_scores = (
                get_norm_cosine_scores(cosine_sim_scores)
                if self.score_type == OutputScore.norm_cosine
                else get_sigmoid_scores(cosine_sim_scores)
            )

            preds = (pos_scores >= self.score_threshold).to(dtype=torch.long)
            scores = torch.cat(
                (neg_scores.unsqueeze(1), pos_scores.unsqueeze(1)), dim=1
            )

        return preds, scores


def get_norm_cosine_scores(cosine_sim_scores):
    # Min-max normalization. It's monotonic in nature and hence doesn't change
    # score distribution.
    pos_scores = (cosine_sim_scores + 1.0) / 2.0
    neg_scores = 1.0 - pos_scores
    return pos_scores, neg_scores


def get_sigmoid_scores(cosine_sim_scores):
    pos_scores = torch.sigmoid(cosine_sim_scores)
    neg_scores = 1.0 - pos_scores
    return pos_scores, neg_scores

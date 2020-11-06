#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from enum import IntEnum, unique
from typing import Any, Dict, List, Optional, Tuple, Union

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
from pytext.utils.usage import log_class_usage


@unique
class OutputScore(IntEnum):
    raw_cosine = 1
    norm_cosine = 2
    sigmoid_cosine = 3


class PairwiseCosineDistanceOutputLayer(OutputLayerBase):
    __EXPANSIBLE__ = True

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
            list(labels) if labels is not None else None,
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
        log_class_usage(__class__)

    def get_loss(
        self,
        logits: Tuple[torch.Tensor, torch.Tensor],
        targets: torch.Tensor,
        context: Optional[Dict[str, Any]] = None,
        reduce: bool = True,
    ) -> torch.Tensor:
        def _transform_logits(logits):
            if isinstance(self.loss_fn, CosineEmbeddingLoss):
                return logits
            elif isinstance(self.loss_fn, NLLLoss):
                # log probability shall be returned in this case
                # the factor 2.3 = log(10) is used to make sure the consistency between matric reporting and loss function
                cosine_sim_scores = F.cosine_similarity(logits[0], logits[1], dim=1)
                pos_scores = cosine_sim_scores * 2.0 * 2.3
                neg_scores = (1.0 - cosine_sim_scores) * 2.0 * 2.3

                return F.log_softmax(
                    torch.stack((neg_scores, pos_scores), dim=1), dim=1
                )
            else:
                return F.cosine_similarity(logits[0], logits[1], dim=1)

        def _transform_targets(targets):
            if isinstance(self.loss_fn, (BinaryCrossEntropyLoss, NLLLoss)):
                return targets
            if isinstance(self.loss_fn, (MAELoss, MSELoss)):
                return targets.to(dtype=torch.float)
            # Replace label = 0 with -1 because we're using cosine_embedding_loss.
            targets = targets.masked_fill(mask=(targets == 0), value=-1.0)
            return targets.to(dtype=torch.float)

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
            scores[:, 0] = (1.0 - cosine_sim_scores[:]) * 2.0
            scores[:, 1] = cosine_sim_scores[:] * 2.0
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


class DenseRetrievalOutputLayer(PairwiseCosineDistanceOutputLayer):
    def get_loss(
        self,
        logits: Tuple[torch.Tensor, torch.Tensor],
        targets: torch.Tensor,
        context: Optional[Dict[str, Any]] = None,
        reduce: bool = True,
    ) -> torch.Tensor:
        # Loss computation pointer: https://fburl.com/inf9ra38
        log_probs = self._get_log_probs(logits)
        # only supports NLL loss
        loss = self.loss_fn(log_probs, targets, reduce)
        return loss

    def get_pred(self, logits: torch.Tensor, targets: torch.Tensor, *args, **kwargs):
        log_probs = self._get_log_probs(logits)
        _, preds = torch.max(log_probs, 1)
        # Pack in logits and positive_indices_per_question for computing Avg Rank
        question_logits, context_logits = logits
        return (
            (
                preds,
                question_logits.detach().cpu().tolist(),
                context_logits.detach().cpu().tolist(),
            ),
            log_probs,
        )  # Expected tuple: prediction, scores

    def _get_log_probs(self, logits):
        # question_logits: (bsz X rep_dim); context_logits: (bsz * 2, rep_dim)
        question_logits, context_logits = logits
        dot_products = torch.matmul(
            question_logits, torch.transpose(context_logits, 0, 1)
        )  # (bsz X bsz*num_negs+1)
        return F.log_softmax(dot_products, dim=1)


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

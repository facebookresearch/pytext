#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


from pytext.config.component import create_loss
from pytext.loss import PairwiseRankingLoss

from .output_layer_base import OutputLayerBase


class PairwiseRankingOutputLayer(OutputLayerBase):
    @classmethod
    def from_config(cls, config):
        return cls(None, create_loss(config.loss), config)

    class Config(OutputLayerBase.Config):  # noqa: T484
        loss: PairwiseRankingLoss.Config = PairwiseRankingLoss.Config()

    def get_pred(self, logit, targets, context):
        pos_similarity, neg_similarity, _sz = PairwiseRankingLoss.get_similarities(
            logit
        )
        preds = pos_similarity > neg_similarity
        scores = pos_similarity - neg_similarity
        return preds, scores

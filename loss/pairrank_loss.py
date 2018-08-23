#!/usr/bin/env python3

import torch.nn.functional as F
from .loss import Loss


class PairRankLoss(Loss):
    def __init__(self, config, margin: float, **kwargs) -> None:
        super().__init__(config)
        self._margin = margin

    def loss(self, m_out, targets, model=None, reduce: bool=True):
        [m_pos], [m_neg] = m_out
        return F.margin_ranking_loss(
            m_pos, m_neg, targets, margin=self._margin, reduce=reduce,
        )

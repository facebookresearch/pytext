#!/usr/bin/env python3

import torch.nn.functional as F

from .loss import Loss


class LanguageModelCrossEntropyLoss(Loss):
    def __init__(self, config, pad_idx, **kwargs):
        super().__init__(config)
        self._ignore_index = pad_idx

    def loss(
        self, m_out, targets, model=None, context=None, reduce: bool = True, weight=None
    ):
        [m_out], [targets] = m_out, targets
        return F.cross_entropy(
            m_out,
            targets.data,
            ignore_index=self._ignore_index,
            reduce=reduce,
            weight=weight,
        )

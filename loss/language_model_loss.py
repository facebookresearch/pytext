#!/usr/bin/env python3

import torch.nn.functional as F
from pytext.common.constants import DatasetFieldName
from pytext.data import CommonMetadata

from .loss import Loss


class LanguageModelCrossEntropyLoss(Loss):
    @classmethod
    def from_config(cls, config, metadata: CommonMetadata, *args, **kwargs):
        return cls(metadata.features[DatasetFieldName.TEXT_FIELD].pad_token_idx)

    def __init__(self, pad_idx):
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

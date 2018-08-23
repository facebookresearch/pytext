#!/usr/bin/env python3

import torch
import torch.nn.functional as F
from pytext.common.constants import Padding

from .loss import Loss


class TaggerCrossEntropyLoss(Loss):
    def loss(self, m_out, targets, model=None, context=None, reduce: bool = True):
        [m_out], [targets] = m_out, targets
        return F.cross_entropy(
            m_out,
            targets.data,
            # TODO: Not a good way of supplying ignore_index. Should rethink
            ignore_index=Padding.WORD_LABEL_PAD_IDX,
            reduce=reduce,
        )


class CRFLoss(Loss):
    def loss(self, m_out, targets, model=None, context=None, reduce: bool = True):
        assert model is not None, "Model cannot be None for CRFLoss"
        [m_out], [targets] = m_out, targets
        loss = -1 * model.crf(
            torch.transpose(m_out, 0, 1).contiguous(),
            torch.transpose(targets, 0, 1).contiguous(),
            reduce=False,
        )
        return loss.mean() if reduce else loss

    def is_crf(self):
        return True

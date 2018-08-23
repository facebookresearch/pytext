#!/usr/bin/env python3

import torch.nn.functional as F
from pytext.config import ConfigBase
from pytext.utils.cuda_utils import FloatTensor

from .loss import Loss


class CrossEntropyLoss(Loss):
    def loss(self, m_out, targets, model=None, context=None, reduce: bool = True):
        [m_out], [targets] = m_out, targets
        return F.cross_entropy(m_out, targets, reduce=reduce)


class BinaryCrossEntropyLoss(Loss):

    class Config(ConfigBase):
        reweight_negative: bool = True

    def loss(self, m_out, targets, model=None, context=None, reduce: bool = True):
        """
        Computes 1-vs-all binary cross entropy loss for multiclass
        classification.
        """
        [m_out], [targets] = m_out, targets
        # Converts targets to one-hot representation. Dim: [batch, n_classes]
        one_hot_targets = (
            FloatTensor(targets.size(0), m_out.size(1))
            .zero_()
            .scatter_(1, targets.unsqueeze(1).data, 1)
        )

        # This weighting applies uniform class weights.
        # examples_per_class = one_hot_target.sum(0).clamp(min=1)
        # total_positive = examples_per_class.sum()
        # weights = total_positive.unsqueeze(0) / examples_per_class

        loss = F.binary_cross_entropy_with_logits(m_out, one_hot_targets, reduce=False)

        if self.config.reweight_negative:
            # This makes sure we have same weights for all negative classes and
            # single positive class. Weight is 1 for the correct class and
            # 1 / (n - 1) for other ones.
            weights = one_hot_targets + (1.0 - one_hot_targets) / max(
                1, one_hot_targets.size(1) - 1.0
            )
            loss = loss * weights

        return loss.sum(1).mean() if reduce else loss.sum(1)

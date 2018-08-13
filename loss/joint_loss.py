#!/usr/bin/env python3

from typing import Union

import torch
from pytext.common.constants import DatasetFieldName
from pytext.common.registry import LOSS, component, create_loss
from pytext.config import ConfigBase

from .classifier_loss import BinaryCrossEntropyLossConfig, CrossEntropyLossConfig
from .loss import Loss
from .tagger_loss import CRFLoss, CRFLossConfig, TaggerCrossEntropyLossConfig


class JointLossConfig(ConfigBase):
    doc_loss: Union[CrossEntropyLossConfig, BinaryCrossEntropyLossConfig]
    word_loss: Union[CRFLossConfig, TaggerCrossEntropyLossConfig]


@component(LOSS, config_cls=JointLossConfig)
class JointLoss(Loss):
    """Base class for joint classification and word tagging loss functions"""

    @classmethod
    def from_config(cls, config: JointLossConfig, **kwargs):
        return cls(create_loss(config.doc_loss), create_loss(config.word_loss))

    def __init__(self, doc_loss: Loss, word_loss: Loss) -> None:
        self._d_loss = doc_loss
        self._w_loss = word_loss

    def loss(self, m_out, targets, model, context, reduce: bool = True):
        d_logit, w_logit = m_out
        d_target, w_target = targets
        d_weight = context[DatasetFieldName.DOC_WEIGHT_FIELD]
        w_weight = context[DatasetFieldName.WORD_WEIGHT_FIELD]
        d_loss = self._d_loss.loss(
            [d_logit], [d_target], context=context, model=model, reduce=False
        )
        w_loss = self._w_loss.loss(
            [w_logit], [w_target], context=context, model=model, reduce=False
        )

        word_per_sentence = w_target.size()[0] // d_target.size()[0]
        w_loss = w_loss.reshape([d_target.size()[0], word_per_sentence])
        w_loss = torch.mean(w_loss, dim=1)
        d_weighted_loss = torch.mean(torch.mul(d_loss, d_weight))
        w_weighted_loss = torch.mean(torch.mul(w_loss, w_weight))
        return d_weighted_loss + w_weighted_loss

    def is_crf(self):
        return isinstance(self._w_loss, CRFLoss)

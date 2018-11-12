#!/usr/bin/env python3
from typing import Union

import torch
from pytext.common.constants import DatasetFieldName
from pytext.config import ConfigBase
from pytext.models.module import create_module
from pytext.data import CommonMetadata

from .output_layer import ClassificationOutputLayer, CRFOutputLayer, OutputLayerBase
from .word_tagging_output_layer import WordTaggingOutputLayer


class IntentSlotOutputLayer(OutputLayerBase):
    class Config(OutputLayerBase.Config):
        doc_output: ClassificationOutputLayer.Config = (
            ClassificationOutputLayer.Config()
        )
        word_output: Union[
            WordTaggingOutputLayer.Config, CRFOutputLayer.Config
        ] = WordTaggingOutputLayer.Config()

    @classmethod
    def from_config(cls, config, meta: CommonMetadata):
        return cls(
            create_module(config.doc_output, meta),
            create_module(config.word_output, meta),
        )

    def __init__(self, doc_output, word_output):
        super().__init__()
        self.doc_output = doc_output
        self.word_output = word_output

    def get_loss(self, logit, target, context):
        d_logit, w_logit = logit
        d_target, w_target = target
        d_weight = context[DatasetFieldName.DOC_WEIGHT_FIELD]
        w_weight = context[DatasetFieldName.WORD_WEIGHT_FIELD]
        d_loss = self.doc_output.get_loss(
            d_logit, d_target, context=context, reduce=False
        )
        w_loss = self.word_output.get_loss(
            w_logit, w_target, context=context, reduce=False
        )
        # w_loss could have been flattened
        if w_loss.size()[0] != w_target.size()[0]:
            w_loss = w_loss.reshape(w_target.size())
            w_loss = torch.mean(w_loss, dim=1)
        d_weighted_loss = torch.mean(torch.mul(d_loss, d_weight))
        w_weighted_loss = torch.mean(torch.mul(w_loss, w_weight))
        return d_weighted_loss + w_weighted_loss

    def get_pred(self, logit, target, context):
        d_logit, w_logit = logit
        if target is not None:
            d_target, w_target = target
        else:
            d_target, w_target = None, None
        d_pred, d_score = self.doc_output.get_pred(d_logit, d_target, context)
        w_pred, w_score = self.word_output.get_pred(w_logit, w_target, context)
        return (d_pred, w_pred), (d_score, w_score)

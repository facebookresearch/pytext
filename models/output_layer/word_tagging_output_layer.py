#!/usr/bin/env python3
import torch
import torch.nn.functional as F
from pytext.common.constants import DatasetFieldName
from pytext.config import ConfigBase
from pytext.config.component import create_loss
from pytext.data import CommonMetadata
from pytext.loss import CrossEntropyLoss

from .output_layer import OutputLayerBase


class WordTaggingOutputLayer(OutputLayerBase):
    class Config(OutputLayerBase.Config, ConfigBase):
        loss: CrossEntropyLoss.Config = CrossEntropyLoss.Config()

    @classmethod
    def from_config(cls, config, meta: CommonMetadata):
        return cls(
            create_loss(
                config.loss,
                ignore_index=meta.labels[
                    DatasetFieldName.WORD_LABEL_FIELD
                ].pad_token_idx,
            )
        )

    def get_loss(self, logit, target, context, reduce=True):
        # flatten the logit from [batch_size, seq_lens, dim] to
        # [batch_size * seq_lens, dim]
        return self.loss_fn(logit.view(-1, logit.size()[-1]), target.view(-1), reduce)

    def get_pred(self, logit, target, context):
        preds = torch.max(logit, 2)[1]
        scores = F.log_softmax(logit, 2)
        return preds, scores

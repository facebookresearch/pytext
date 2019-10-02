#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn.functional as F
from pytext.config.component import create_loss
from pytext.loss import CrossEntropyLoss
from pytext.models.output_layers.output_layer_base import OutputLayerBase


class MyTaggingOutputLayer(OutputLayerBase):
    class Config(OutputLayerBase.Config):
        loss: CrossEntropyLoss.Config = CrossEntropyLoss.Config()

    @classmethod
    def from_config(cls, config, vocab, pad_token):
        return cls(vocab, create_loss(config.loss, ignore_index=pad_token))

    def get_loss(self, logit, target, context, reduce=True):
        # flatten the logit from [batch_size, seq_lens, dim] to
        # [batch_size * seq_lens, dim]
        return self.loss_fn(logit.view(-1, logit.size()[-1]), target.view(-1), reduce)

    def get_pred(self, logit, *args, **kwargs):
        preds = torch.max(logit, 2)[1]
        scores = F.log_softmax(logit, 2)
        return preds, scores

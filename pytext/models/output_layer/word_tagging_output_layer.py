#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List

import torch
import torch.nn.functional as F
from caffe2.python import core
from pytext.config.component import create_loss
from pytext.fields import FieldMeta
from pytext.loss import CrossEntropyLoss

from .output_layer import OutputLayerBase, gen_additional_blobs


class WordTaggingOutputLayer(OutputLayerBase):
    class Config(OutputLayerBase.Config):
        loss: CrossEntropyLoss.Config = CrossEntropyLoss.Config()

    @classmethod
    def from_config(cls, config, meta: FieldMeta):
        return cls(
            meta.vocab.itos, create_loss(config.loss, ignore_index=meta.pad_token_idx)
        )

    def get_loss(self, logit, target, context, reduce=True):
        # flatten the logit from [batch_size, seq_lens, dim] to
        # [batch_size * seq_lens, dim]
        return self.loss_fn(logit.view(-1, logit.size()[-1]), target.view(-1), reduce)

    def get_pred(self, logit, target, context):
        preds = torch.max(logit, 2)[1]
        scores = F.log_softmax(logit, 2)
        return preds, scores

    def export_to_caffe2(
        self,
        workspace: core.workspace,
        init_net: core.Net,
        predict_net: core.Net,
        model_out: torch.Tensor,
        output_name: str,
    ) -> List[core.BlobReference]:
        """
        Exports the intent slot output layer to caffe2 by manually adding the
        necessary operators to the init_net and predict net, and returns the
        list of external output blobs to be added to the model.
        """
        probability_out = predict_net.Softmax(output_name, axis=model_out.dim() - 1)
        return gen_additional_blobs(
            predict_net, probability_out, model_out, output_name, self.target_names
        )

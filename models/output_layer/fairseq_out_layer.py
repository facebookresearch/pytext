#!/usr/bin/env python3
from pytext.common.constants import DatasetFieldName
from pytext.config import ConfigBase
from pytext.config.component import create_loss
from pytext.data import CommonMetadata
from pytext.loss import CrossEntropyLoss

from .output_layer import OutputLayerBase


class FairseqOutputLayer(OutputLayerBase):
    class Config(ConfigBase):
        loss: CrossEntropyLoss.Config = CrossEntropyLoss.Config()

    @classmethod
    def from_config(cls, config, meta: CommonMetadata):
        return cls(
            create_loss(
                config.loss,
                meta.labels[DatasetFieldName.TARGET_SEQ_FIELD].pad_token_idx,
            )
        )

    def get_loss(self, logit, target, context, reduce=True):
        # flatten the logit from [batch_size, seq_lens, dim] to
        # [batch_size * seq_lens, dim]
        return self.loss_fn(
            logit[0].view(-1, logit[0].size()[-1]), target[0].view(-1), reduce
        )

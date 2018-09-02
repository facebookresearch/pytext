#!/usr/bin/env python3
from pytext.config import ConfigBase
from pytext.loss import CrossEntropyLoss
from pytext.config.component import create_loss
from pytext.common.constants import DatasetFieldName
from pytext.data import CommonMetadata
from .output_layer import OutputLayerBase


class LMOutputLayer(OutputLayerBase):
    class Config(ConfigBase):
        loss: CrossEntropyLoss.Config = CrossEntropyLoss.Config()

    @classmethod
    def from_config(cls, config, meta: CommonMetadata):
        return cls(
            create_loss(
                config.loss,
                meta.features[DatasetFieldName.TEXT_FIELD].pad_token_idx,
            )
        )

    def get_loss(self, logit, target, context, reduce=True):
        # flatten the logit from [batch_size, seq_lens, dim] to
        # [batch_size * seq_lens, dim]
        return self.loss_fn(logit.view(-1, logit.size()[-1]), target.view(-1), reduce)

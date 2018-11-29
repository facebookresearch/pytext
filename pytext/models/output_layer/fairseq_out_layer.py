#!/usr/bin/env python3
from pytext.config import ConfigBase
from pytext.config.component import create_loss
from pytext.fields import FieldMeta
from pytext.loss import CrossEntropyLoss

from .output_layer import OutputLayerBase


class FairseqOutputLayer(OutputLayerBase):
    class Config(ConfigBase):
        loss: CrossEntropyLoss.Config = CrossEntropyLoss.Config()

    @classmethod
    def from_config(cls, config, meta: FieldMeta):
        return cls(meta.vocab.itos, create_loss(config.loss, meta.pad_token_idx))

    def get_loss(self, logits_tuple, targets_tuple, context, reduce=True):
        # flatten the logit from [batch_size, seq_lens, dim] to
        # [batch_size * seq_lens, dim]
        logits, _attention_scores = logits_tuple
        targets, _target_lens = targets_tuple
        return self.loss_fn(
            logits.view(-1, logits.size()[-1]), targets.view(-1), reduce
        )

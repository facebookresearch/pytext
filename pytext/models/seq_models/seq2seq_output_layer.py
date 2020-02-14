#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Any, Dict, Tuple, Union

import torch
from pytext.config import ConfigBase
from pytext.config.component import create_loss
from pytext.data.utils import Vocabulary
from pytext.loss import CrossEntropyLoss, LabelSmoothedCrossEntropyLoss, NLLLoss
from pytext.models.output_layers import OutputLayerBase


class Seq2SeqOutputLayer(OutputLayerBase):
    class Config(ConfigBase):
        loss: Union[
            CrossEntropyLoss.Config,
            LabelSmoothedCrossEntropyLoss.Config,
            NLLLoss.Config,
        ] = CrossEntropyLoss.Config()

    @classmethod
    def from_config(cls, config: Config, vocab: Vocabulary):
        return cls(vocab._vocab, create_loss(config.loss, vocab.get_pad_index()))

    def get_loss(
        self,
        model_outputs: Tuple[torch.Tensor, Dict[str, torch.Tensor]],
        targets: Tuple[torch.Tensor, torch.Tensor],
        context: Dict[str, Any] = None,
        reduce=True,
    ) -> torch.Tensor:
        # flatten the logit from [batch_size, seq_lens, dim] to
        # [batch_size * seq_lens, dim]
        logits = model_outputs[0]
        loss = self.loss_fn(
            logits.view(-1, logits.size()[-1]), targets[0].view(-1), reduce
        )
        return loss

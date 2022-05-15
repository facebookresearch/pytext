#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Any, Dict, Tuple, Union

import torch
from pytext.config import ConfigBase
from pytext.config.component import create_loss
from pytext.data.utils import Vocabulary
from pytext.loss import NARSamplewiseSequenceLoss, NARSequenceLoss, StructuredLoss
from pytext.models.output_layers import OutputLayerBase


class NARSeq2SeqOutputLayer(OutputLayerBase):
    """Non-autoregressive seq2seq output layer."""

    class Config(ConfigBase):
        loss: Union[
            NARSequenceLoss.Config, NARSamplewiseSequenceLoss.Config
        ] = NARSequenceLoss.Config()

    @classmethod
    def from_config(cls, config: Config, vocab: Vocabulary):
        return cls(
            vocab._vocab, create_loss(config.loss, ignore_index=vocab.get_pad_index())
        )

    def get_loss(
        self,
        model_outputs: Tuple[torch.Tensor, Dict[str, torch.Tensor]],
        targets: Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
        context: Dict[str, Any] = None,
        reduce=True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        label_logits: B x T x V_1
        label_targets: B x T
        length_logits: B x V_2
        length_targets: B
        """

        label_logits, output_dict = model_outputs
        length_logits = output_dict["predicted_tgt_lengths"]
        (_, label_targets), length_targets = targets

        # Structured losses require access to sequences in each batch, so don't
        # flatten logits and targets for these.
        if not isinstance(self.loss_fn.label_loss_fn.label_loss_fn, StructuredLoss):
            label_logits = label_logits.view(-1, label_logits.size(-1))  # (B x T) x V
            label_targets = label_targets.view(-1)  # (B x T)

        loss, two_losses = self.loss_fn(
            label_logits,
            label_targets,
            length_logits,
            length_targets,
            reduce,
        )
        return loss, two_losses

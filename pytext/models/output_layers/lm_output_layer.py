#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from pytext.config.component import create_loss
from pytext.data.utils import PAD, Vocabulary
from pytext.fields import FieldMeta
from pytext.loss import CrossEntropyLoss, Loss

from .output_layer_base import OutputLayerBase


class LMOutputLayer(OutputLayerBase):
    """
    Output layer for language models. It supports `CrossEntropyLoss` per word.

    Args:
        loss_fn (CrossEntropyLoss): Cross-entropy loss component. Defaults to None.

    Attributes:
        loss_fn: Cross-entropy loss component for computing loss.

    """

    class Config(OutputLayerBase.Config):
        loss: CrossEntropyLoss.Config = CrossEntropyLoss.Config()

    @classmethod
    def from_config(
        cls,
        config: Config,
        metadata: Optional[FieldMeta] = None,
        labels: Optional[Vocabulary] = None,
    ):
        if labels is not None:
            vocab = list(labels)
            pad_token_idx = labels.get_pad_index()
        else:
            vocab = metadata.vocab.itos
            pad_token_idx = metadata.pad_token_idx
        return cls(
            vocab,
            create_loss(config.loss, ignore_index=pad_token_idx),
            pad_token_idx=pad_token_idx,
        )

    def __init__(
        self,
        target_names: List[str],
        loss_fn: Loss = None,
        config=None,
        pad_token_idx=-100,
    ):
        super().__init__(target_names, loss_fn, config)
        self.pad_token_idx = pad_token_idx

    def get_loss(
        self,
        logit: torch.Tensor,
        target: torch.Tensor,
        context: Dict[str, Any],
        reduce=True,
    ) -> torch.Tensor:
        """Compute word prediction loss by comparing prediction of each word in the
        sentence with the true word.

        Args:
            logit (torch.Tensor): Logit returned by
                :class:`~pytext.models.language_models.lmlstm.LMLSTM`.
            targets (torch.Tensor): Not applicable for language models.
            context (Dict[str, Any]): Not applicable. Defaults to None.
            reduce (bool): Whether to reduce loss over the batch. Defaults to True.

        Returns:
            torch.Tensor: Word prediction loss.

        """
        if isinstance(target, tuple):
            target = target[0]
        # flatten the logit from [batch_size, seq_lens, dim] to
        # [batch_size * seq_lens, dim]
        return self.loss_fn(logit.view(-1, logit.size()[-1]), target.view(-1), reduce)

    def get_pred(
        self, logit: torch.Tensor, *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute and return prediction and scores from the model.
        Prediction is computed using argmax over the word label/target space.
        Scores are softmax scores over the model logits.

        Args:
            logit (torch.Tensor): Logits returned
                :class:`~pytext.models.language_models.lmlstm.LMLSTM`.
            targets (torch.Tensor): True words.
            context (Dict[str, Any]): Context is a dictionary of items
                that's passed as additional metadata by the
                :class:`~pytext.data.LanguageModelDataHandler`

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Model prediction and scores.

        """
        preds = torch.max(logit, 2)[1]
        scores = F.log_softmax(logit, 2)
        return preds, scores

    @staticmethod
    def calculate_perplexity(sequence_loss: torch.Tensor) -> torch.Tensor:
        return torch.exp(sequence_loss)

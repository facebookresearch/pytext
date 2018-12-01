#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from pytext.common.constants import DatasetFieldName
from pytext.config.component import create_loss
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
    def from_config(cls, config: Config, metadata: FieldMeta):
        return cls(
            metadata.vocab.itos,
            create_loss(config.loss, ignore_index=metadata.pad_token_idx),
            pad_token_idx=metadata.pad_token_idx,
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
        # flatten the logit from [batch_size, seq_lens, dim] to
        # [batch_size * seq_lens, dim]
        return self.loss_fn(logit.view(-1, logit.size()[-1]), target.view(-1), reduce)

    def get_pred(
        self, logit: torch.Tensor, target: torch.Tensor, context: Dict[str, Any]
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
                :class:`~pytext.data.LanguageModelDataHandler` or
                :class:`~pytext.data.BPTTLanguageModelDataHandler`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Model prediction and scores.

        """
        # Shape of logit: (bsize x seq_len x vocab)
        # Reshape m_out to (bsize x vocab x seq_len) for cross_entropy_loss
        logit = logit.transpose(1, 2)
        # loss dim: (bsize x seq_len)
        loss = F.cross_entropy(
            logit, target, reduce=False, ignore_index=self.pad_token_idx
        )
        # context[DatasetFieldName.SEQ_LENS] s the length of each sequence
        # sequence_loss is the loss per word for each sequence in the batch
        # sequence_loss dim: (bsize,)
        sequence_loss = loss.sum(1) / context[DatasetFieldName.TARGET_SEQ_LENS].float()
        scores = self.calculate_perplexity(sequence_loss)
        return scores, scores

    @staticmethod
    def calculate_perplexity(sequence_loss: torch.Tensor) -> torch.Tensor:
        return torch.exp(sequence_loss)

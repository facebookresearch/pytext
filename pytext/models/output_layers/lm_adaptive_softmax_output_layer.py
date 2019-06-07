#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Any, Dict, List, Optional, Tuple

import torch
from pytext.data.utils import PAD, Vocabulary
from pytext.fields import FieldMeta
from torch.nn.modules.adaptive import AdaptiveLogSoftmaxWithLoss

from .output_layer_base import OutputLayerBase


class LMASOutputLayer(OutputLayerBase):
    """
    Output layer for language models. It supports `CrossEntropyLoss` per word.

    Args:
        cutoffs (List[int]): Sequence of integers sorted in the increasing order
                             It controls number of clusters and
                             the partitioning of targets into clusters.
        div_value (float): compute the size of each additional cluster
        head_bias (bool): True adds a bias term to the 'head' of the adaptive softmax

    Attributes:
        ada_logsoftmax : Adaptive LogSoftmax With Loss component
    """

    class Config(OutputLayerBase.Config):
        cutoffs: List[int]
        div_value: float = 4.0
        head_bias: bool = False

    @classmethod
    def from_config(
        cls,
        config: Config,
        metadata: Optional[FieldMeta] = None,
        labels: Optional[Vocabulary] = None,
    ):
        assert labels is not None
        vocab = list(labels)
        pad_token_idx = labels.idx[PAD]
        return cls(
            in_features=len(vocab),
            cutoffs=config.cutoffs,
            target_names=vocab,
            div_value=config.div_value,
            head_bias=config.head_bias,
            pad_token_idx=pad_token_idx,
        )

    def __init__(
        self,
        in_features: int,
        cutoffs: List[int],
        target_names: List[str],
        div_value: float,
        head_bias: bool,
        pad_token_idx: int,
    ):
        super().__init__(target_names)
        self.in_features = in_features
        self.cutoffs = cutoffs
        self.target_names = target_names
        self.div_value = div_value
        self.head_bias = head_bias
        self.pad_token_idx = pad_token_idx
        self.ada_logsoftmax = AdaptiveLogSoftmaxWithLoss(
            self.in_features,
            len(self.target_names),
            self.cutoffs,
            self.div_value,
            self.head_bias,
        )

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

        if logit.size(0) != target.size(0):
            raise RuntimeError(
                "Input and target should have the same size " "in the batch dimension."
            )
        if logit.size(1) != target.size(1):
            raise RuntimeError(
                "Input and target should have the same sequence size "
                "in the batch dimension."
            )
        output = logit.new_zeros(logit.size(0), logit.size(1))
        tokens_num = logit.new_zeros(logit.size(0), logit.size(1))

        for j in range(logit.size(1)):
            target_mask_pad = (target[:, j] != self.pad_token_idx).float()
            output[:, j] = (
                self.ada_logsoftmax.forward(logit[:, j, :], target[:, j])[0]
                * target_mask_pad
            )
            tokens_num[:, j] = target_mask_pad

        if reduce:
            loss = (-torch.sum(output)) / torch.sum(tokens_num)
        else:
            loss = torch.tensor(
                [
                    (-torch.sum(output[i, :])) / torch.sum(tokens_num[i, :])
                    for i in range(output.size(0))
                ]
            )
        return loss

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
                :class:`~pytext.data.LanguageModelDataHandler` or
                :class:`~pytext.data.BPTTLanguageModelDataHandler`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Model prediction and scores.

        """

        preds = logit.new_zeros(logit.size(0), logit.size(1))
        scores = logit.new_empty(logit.size(0), logit.size(1), logit.size(2))

        for i in range(logit.size(1)):
            scores[:, i, :] = self.ada_logsoftmax.log_prob(logit[:, i, :])
            preds[:, i] = torch.argmax(scores[:, i, :], dim=1)
        return preds, scores

    @staticmethod
    def calculate_perplexity(sequence_loss: torch.Tensor) -> torch.Tensor:
        return torch.exp(sequence_loss)

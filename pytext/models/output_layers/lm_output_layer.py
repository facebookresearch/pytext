#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from caffe2.python import core
from pytext.config.component import create_loss
from pytext.data.utils import PAD, Vocabulary
from pytext.fields import FieldMeta
from pytext.loss import CrossEntropyLoss, Loss

from .output_layer_base import OutputLayerBase
from .utils import OutputLayerUtils


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
        self, logits: torch.Tensor, *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute and return prediction and scores from the model.
        Prediction is computed using argmax over the word label/target space.
        Scores are softmax scores over the model logits.

        Args:
            logits (torch.Tensor): Logits returned
                :class:`~pytext.models.language_models.lmlstm.LMLSTM`.
            targets (torch.Tensor): True words.
            context (Dict[str, Any]): Context is a dictionary of items
                that's passed as additional metadata by the
                :class:`~pytext.data.LanguageModelDataHandler`

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Model prediction and scores.

        """
        logits = logits.permute(0, 2, 1)  # [bsz, vocab, seq_len]
        return ((F.log_softmax(logits, 1), self.pad_token_idx), None)

    def export_to_caffe2(
        self,
        workspace: core.workspace,
        init_net: core.Net,
        predict_net: core.Net,
        model_out: torch.Tensor,
        output_name: str,
    ) -> List[core.BlobReference]:
        prob_out = predict_net.Softmax(output_name, axis=model_out.dim() - 1)
        # prepend an underscore to target_names to avoid conflicts between
        # existing cell names and target names
        edited_target_names = [f"_{name}" for name in self.target_names]
        return OutputLayerUtils.gen_additional_blobs(
            predict_net, prob_out, model_out, output_name, edited_target_names
        )

    @staticmethod
    def calculate_perplexity(sequence_loss: torch.Tensor) -> torch.Tensor:
        return torch.exp(sequence_loss)

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from caffe2.python import core
from pytext.common.constants import DatasetFieldName
from pytext.data.utils import Vocabulary
from pytext.models.module import create_module
from torch import jit

from .doc_classification_output_layer import ClassificationOutputLayer
from .output_layer_base import OutputLayerBase
from .word_tagging_output_layer import CRFOutputLayer, WordTaggingOutputLayer


class IntentSlotScores(nn.Module):
    def __init__(self, doc_scores: jit.ScriptModule, word_scores: jit.ScriptModule):
        super().__init__()
        self.doc_scores = doc_scores
        self.word_scores = word_scores

    def forward(
        self,
        logits: Tuple[torch.Tensor, torch.Tensor],
        context: Dict[str, torch.Tensor],
    ) -> Tuple[List[Dict[str, float]], List[List[Dict[str, float]]]]:
        d_logits, w_logits = logits
        if "token_indices" in context:
            w_logits = torch.gather(
                w_logits,
                1,
                context["token_indices"].unsqueeze(2).expand(-1, -1, w_logits.size(-1)),
            )

        d_results = self.doc_scores(d_logits)
        w_results = self.word_scores(w_logits, context)
        return d_results, w_results


class IntentSlotOutputLayer(OutputLayerBase):
    """
    Output layer for joint intent classification and slot-filling models.
    Intent classification is a document classification problem and slot filling
    is a word tagging problem. Thus terms these can be used interchangeably in the
    documentation.

    Args:
        doc_output (ClassificationOutputLayer): Output layer for intent
            classification task. See
            :class:`~pytext.models.output_layers.ClassificationOutputLayer` for
            details.
        word_output (WordTaggingOutputLayer): Output layer for slot filling task.
            See :class:`~pytext.models.output_layers.WordTaggingOutputLayer` for
            details.

    Attributes:
        doc_output (type): Output layer for intent classification task.
        word_output (type): Output layer for slot filling task.

    """

    class Config(OutputLayerBase.Config):
        doc_output: ClassificationOutputLayer.Config = (
            ClassificationOutputLayer.Config()
        )
        word_output: Union[
            WordTaggingOutputLayer.Config, CRFOutputLayer.Config
        ] = WordTaggingOutputLayer.Config()

    @classmethod
    def from_config(
        cls, config: Config, doc_labels: Vocabulary, word_labels: Vocabulary
    ):
        return cls(
            create_module(config.doc_output, labels=doc_labels),
            create_module(config.word_output, labels=word_labels),
        )

    def __init__(
        self, doc_output: ClassificationOutputLayer, word_output: WordTaggingOutputLayer
    ) -> None:
        super().__init__()
        self.doc_output = doc_output
        self.word_output = word_output

    def get_loss(
        self,
        logits: Tuple[torch.Tensor, torch.Tensor],
        targets: Tuple[torch.Tensor, torch.Tensor],
        context: Dict[str, Any] = None,
        *args,
        **kwargs
    ) -> torch.Tensor:
        """Compute and return the averaged intent and slot-filling loss.

        Args:
            logit (Tuple[torch.Tensor, torch.Tensor]): Logits returned by
                :class:`~pytext.models.joint_model.JointModel`. It is a tuple
                containing logits for intent classification and slot filling.
            targets (Tuple[torch.Tensor, torch.Tensor]): Tuple of target Tensors
                containing true document label/target and true word labels/targets.
            context (Dict[str, Any]): Context is a dictionary of items
                that's passed as additional metadata. Defaults to None.

        Returns:
            torch.Tensor: Averaged intent and slot loss.

        """
        d_logit, w_logit = logits
        if DatasetFieldName.TOKEN_INDICES in context:
            w_logit = torch.gather(
                w_logit,
                1,
                context[DatasetFieldName.TOKEN_INDICES]
                .unsqueeze(2)
                .expand(-1, -1, w_logit.size(-1)),
            )
        d_target, w_target = targets
        d_weight = context[DatasetFieldName.DOC_WEIGHT_FIELD]  # noqa
        w_weight = context[DatasetFieldName.WORD_WEIGHT_FIELD]  # noqa
        d_loss = self.doc_output.get_loss(
            d_logit, d_target, context=context, reduce=False
        )
        w_loss = self.word_output.get_loss(
            w_logit, w_target, context=context, reduce=False
        )
        # w_loss could have been flattened
        w_hard_target = w_target[0] if type(w_target) is tuple else w_target
        if w_loss.size()[0] != w_hard_target.size()[0]:
            w_loss = w_loss.reshape(w_hard_target.size())
            w_loss = torch.mean(w_loss, dim=1)
        d_weighted_loss = torch.mean(torch.mul(d_loss, d_weight))
        w_weighted_loss = torch.mean(torch.mul(w_loss, w_weight))
        return d_weighted_loss + w_weighted_loss

    def get_pred(
        self,
        logits: Tuple[torch.Tensor, torch.Tensor],
        targets: Optional[torch.Tensor] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute and return prediction and scores from the model.

        Args:
            logit (Tuple[torch.Tensor, torch.Tensor]): Logits returned by
                :class:`~pytext.models.joint_model.JointModel`. It's tuple
                containing logits for intent classification and slot filling.
            targets (Optional[torch.Tensor]): Not applicable. Defaults to None.
            context (Optional[Dict[str, Any]]): Context is a dictionary of items
                that's passed as additional metadata. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Model prediction and scores.

        """
        d_logit, w_logit = logits
        if DatasetFieldName.TOKEN_INDICES in context:
            w_logit = torch.gather(
                w_logit,
                1,
                context[DatasetFieldName.TOKEN_INDICES]
                .unsqueeze(2)
                .expand(-1, -1, w_logit.size(-1)),
            )
        d_pred, d_score = self.doc_output.get_pred(d_logit, None, context)
        w_pred, w_score = self.word_output.get_pred(w_logit, None, context)
        return (d_pred, w_pred), (d_score, w_score)

    def export_to_caffe2(
        self,
        workspace: core.workspace,
        init_net: core.Net,
        predict_net: core.Net,
        model_out: List[torch.Tensor],
        doc_out_name: str,
        word_out_name: str,
    ) -> List[core.BlobReference]:
        """
        Exports the intent slot output layer to Caffe2.
        See `OutputLayerBase.export_to_caffe2()` for details.
        """
        return self.doc_output.export_to_caffe2(
            workspace, init_net, predict_net, model_out[0], doc_out_name
        ) + self.word_output.export_to_caffe2(
            workspace, init_net, predict_net, model_out[1], word_out_name
        )

    def torchscript_predictions(self):
        doc_scores = self.doc_output.torchscript_predictions()
        word_scores = self.word_output.torchscript_predictions()
        return jit.script(IntentSlotScores(doc_scores, word_scores))

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import functools
import operator
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from caffe2.python import core
from pytext.data.tensorizers import Tensorizer
from pytext.models.module import create_module
from pytext.utils.usage import log_class_usage
from torch import jit

from .doc_classification_output_layer import ClassificationOutputLayer
from .output_layer_base import OutputLayerBase


class MultiLabelClassificationScores(nn.Module):
    def __init__(self, scores: jit.ScriptModule):
        super().__init__()
        self.scores = scores
        log_class_usage(__class__)

    def forward(
        self, logits: List[torch.Tensor]
    ) -> Tuple[List[List[Dict[str, float]]]]:

        results = []
        for logit_ in logits:
            results.append(self.scores(logit_))
        return results


class MultiLabelClassificationLayer(OutputLayerBase):
    """
    Output layer for multilabel sequence classification models.

    Args:
        output (List[WordTaggingOutputLayer]): Output for multilabels, here
            USM + PTSR + EA task.

        label_names (List[str]): Ordered list of labels predicted through the model
            for which the losses need to be aggregated by the output layer

        label_tensorizer (Dict[str, LabelListTensorizer]): Dict of list of labels
            that constitute the output from the decoder ordered by label_names
            sequencing

        optional label_weights (Dict[str, int]): Dict of label_names along with the
        weight for label

    Attributes:
        output (type): Output layer for multilabel-multiclass classification task
        label_names (type): List of labels to be predicted by the model
        label_tensorizer (type): Dict of key-label names with values-tensorizers
        used to compute the size of the label vocab
        optional label_weights (type): Dict of label-weight to compute weighted
        output layer

    """

    class Config(OutputLayerBase.Config):
        output: List[ClassificationOutputLayer.Config] = []
        label_weights: Dict[str, float] = {}

    @classmethod
    def from_config(
        cls,
        config: Config,
        label_tensorizers: [Dict[str, Tensorizer]],
        label_names: [List[str]],
    ):
        modules = []
        for label_idx in range(0, len(label_names)):
            label_ = label_names[label_idx]
            modules.append(
                create_module(
                    config.output[label_idx], labels=label_tensorizers[label_].vocab
                )
            )
        print("Created Modules", len(modules))
        return cls(modules, label_names, config.label_weights)

    def __init__(
        self,
        output: List[ClassificationOutputLayer],
        label_names: List[str],
        label_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        super().__init__()
        self.output = output
        self.label_names = label_names
        self.label_weights = label_weights
        log_class_usage(__class__)

    def get_loss(
        self,
        logits,
        targets: List[torch.Tensor],
        context: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs
    ) -> torch.Tensor:
        """Compute and return the averaged intent and slot-filling loss.

        Args:
            logits (List[tuple[torch.Tensor]]): Logits returned by
                :class:`~pytext.models.decoders.MultiLabelDecoder`. It's list
                containing logits for all label tasks here pTSR, autoUSM and EA.
            targets (List[tuple[torch.Tensor]]): Targets as computed by the true labels
            optional label_weights (Dict[str, float]): Label weights for multi-label
            ordering of logits corresponding the respective label.
            targets (Optional[torch.Tensor]): Not applicable. Defaults to None.

        Returns:
            torch.Tensor: Averaged Loss across all label losses.
        """
        loss = 0
        for label_idx, label_name in enumerate(self.label_names):
            logit = logits[label_idx]
            # [batch_size * seq_lens, dim]
            flattened_logit = logit.view(-1, logit.size()[-1])
            loss += self.output[label_idx].get_loss(
                flattened_logit, targets[label_idx].view(-1), None
            )
            if self.label_weights:
                weight = self.label_weights[label_name]
                loss = torch.mean(torch.mul(loss, weight))
        loss = loss / (len(self.label_names) * 1.0)
        return loss

    def get_pred(
        self,
        logits: List[torch.Tensor],
        targets: List[torch.Tensor],
        context: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute and return prediction and scores from the model.

        Prediction is computed using argmax over the document label/target space.

        Scores are sigmoid or softmax scores over the model logits depending on
        the loss component being used.

        Args:
            logits (List[torch.Tensor]): Logits returned by
                :class:`~pytext.models.decoders.MultiLabelDecoder`. It's list
                containing logits for all label tasks here pTSR, autoUSM and EA.
            targets (List[tuple[torch.Tensor]]): Targets as computed by the true labels
            ordering of logits corresponding the respective label.
            targets (Optional[torch.Tensor]): Not applicable. Defaults to None.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Model prediction and scores.

        """
        # The assumption here is that the logit is a concat of the
        # label predictions ordered by label list.
        scores = []
        preds = []
        for label_idx, logit in enumerate(logits):
            pred, score = self.output[label_idx].get_pred(logit)
            preds.append(pred)
            scores.append(score)
        return (preds, scores)

    def export_to_caffe2(
        self,
        workspace: core.workspace,
        init_net: core.Net,
        predict_net: core.Net,
        model_out: List[torch.Tensor],
        out_name: str,
    ) -> List[core.BlobReference]:
        """
        Exports the multilabel output layer to Caffe2.
        See `OutputLayerBase.export_to_caffe2()` for details.
        """

        return functools.reduce(
            operator.add,
            [
                self.output[idx].export_to_caffe2(
                    workspace, init_net, predict_net, model_out[idx], out_name
                )
                for idx, single_output in enumerate(model_out)
            ],
        )

    def torchscript_predictions(self):
        scores = self.output.torchscript_predictions()
        return jit.script(MultiLabelClassificationScores(scores))

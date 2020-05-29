#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import functools
import operator
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from caffe2.python import core
from pytext.data.tensorizers import Tensorizer
from pytext.utils.usage import log_class_usage
from torch import jit

from .doc_classification_output_layer import ClassificationOutputLayer
from .output_layer_base import OutputLayerBase


class MultiLabelClassificationScores(nn.Module):
    def __init__(self, scores: List[jit.ScriptModule]):
        super().__init__()
        self.scores = nn.ModuleList(scores)
        log_class_usage(__class__)

    def forward(self, logits: List[torch.Tensor]) -> List[List[Dict[str, float]]]:

        results: List[List[Dict[str, float]]] = []
        for idx, sc in enumerate(self.scores):
            logit = logits[idx]
            # flatten from [batch_size, ..., label_set_size] to
            # [batch_size, label_set_size]
            # must flatten because jit doesn't support dynamic return type
            flattened_logit = logit.view(-1, logit.size()[-1])
            results.append(sc(flattened_logit))

        return results


class MultiLabelClassificationLayer(OutputLayerBase):
    """
    Output layer for multilabel sequence classification models.

    Args:
        outputs (Dict[str, ClassificationOutputLayer]): Output for multilabels
        optional label_weights (Dict[str, int]): Dict of label_names along with the
        weight for label
    """

    class Config(OutputLayerBase.Config):
        outputs: List[ClassificationOutputLayer.Config] = []
        label_set_weights: Dict[str, float] = {}

    @classmethod
    def from_config(cls, config: Config, label_tensorizers: [Dict[str, Tensorizer]]):
        modules = {
            name: ClassificationOutputLayer.from_config(
                config.outputs[idx], labels=tensorizer.vocab
            )
            for idx, (name, tensorizer) in enumerate(label_tensorizers.items())
        }

        return cls(modules, config.label_set_weights)

    def __init__(
        self,
        outputs: Dict[str, ClassificationOutputLayer],
        label_set_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        super().__init__()
        self.outputs = outputs
        self.num_label_sets = len(outputs)
        self.label_set_weights = label_set_weights
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
            logits (List[torch.Tensor]): Logits returned by
                :class:`~pytext.models.decoders.MultiLabelDecoder`. It's list
                containing logits for all label tasks here pTSR, autoUSM and EA.
            targets (List[torch.Tensor]): Targets as computed by the true labels
            context (Optional[torch.Tensor]): Not applicable. Defaults to None.

        Returns:
            torch.Tensor: Averaged Loss across all label losses.
        """
        total_loss = 0
        for logit, target, (label_name, output_layer) in zip(
            logits, targets, self.outputs.items()
        ):
            # flatten from [batch_size, ..., label_set_size] to
            # [batch_size, label_set_size]
            flattened_logit = logit.view(-1, logit.size()[-1])
            loss = output_layer.get_loss(flattened_logit, target.view(-1))
            if label_name in self.label_set_weights:
                weight = self.label_set_weights[label_name]
                loss = torch.mul(loss, weight)
            total_loss += loss
        return total_loss / (self.num_label_sets * 1.0)

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
            targets (List[torch.Tensor]): Targets as computed by the true labels
            ordering of logits corresponding the respective label.
            context (Optional[torch.Tensor]): Not applicable. Defaults to None.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Model prediction and scores.

        """
        # The assumption here is that the logit is a concat of the
        # label predictions ordered by label list.
        scores = []
        preds = []
        for output_layer, logit in zip(self.outputs.values(), logits):
            pred, score = output_layer.get_pred(logit)
            preds.append(pred)
            scores.append(score)
        return (tuple(preds), tuple(scores))

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
                output_layer.export_to_caffe2(
                    workspace, init_net, predict_net, single_output, out_name
                )
                for output_layer, single_output in zip(self.outputs, model_out)
            ],
        )

    def torchscript_predictions(self):
        scores = [o.torchscript_predictions() for o in self.outputs.values()]
        return jit.script(MultiLabelClassificationScores(scores))

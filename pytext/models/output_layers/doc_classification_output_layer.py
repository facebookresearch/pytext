#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from caffe2.python import core
from pytext.config.component import create_loss
from pytext.data.utils import Vocabulary
from pytext.fields import FieldMeta
from pytext.loss import (
    AUCPRHingeLoss,
    BinaryCrossEntropyLoss,
    CrossEntropyLoss,
    KLDivergenceBCELoss,
    KLDivergenceCELoss,
    SoftHardBCELoss,
)
from pytext.utils.label import get_label_weights
from torch import jit

from .output_layer_base import OutputLayerBase
from .utils import OutputLayerUtils


class ClassificationOutputLayer(OutputLayerBase):
    """
    Output layer for document classification models.
    It supports `CrossEntropyLoss` and `BinaryCrossEntropyLoss` per document.

    Args:
        loss_fn (Union[CrossEntropyLoss, BinaryCrossEntropyLoss]):
            The loss function to use for computing loss. Defaults to None.

    Attributes:
        loss_fn: The loss function to use for computing loss.

    """

    class Config(OutputLayerBase.Config):
        loss: Union[
            CrossEntropyLoss.Config,
            BinaryCrossEntropyLoss.Config,
            AUCPRHingeLoss.Config,
            KLDivergenceBCELoss.Config,
            KLDivergenceCELoss.Config,
            SoftHardBCELoss.Config,
        ] = CrossEntropyLoss.Config()
        label_weights: Optional[Dict[str, float]] = None

    @classmethod
    def from_config(
        cls,
        config: Config,
        metadata: Optional[FieldMeta] = None,
        labels: Optional[Vocabulary] = None,
    ):
        if labels is not None:
            vocab = list(labels)
            vocab_dict = labels.idx
        else:
            vocab = metadata.vocab.itos
            vocab_dict = metadata.vocab.stoi

        label_weights = (
            get_label_weights(vocab_dict, config.label_weights)
            if config.label_weights
            else None
        )
        loss = create_loss(config.loss, weight=label_weights)
        cls = (
            BinaryClassificationOutputLayer
            if isinstance(loss, BinaryCrossEntropyLoss)
            else MulticlassOutputLayer
        )
        return cls(vocab, create_loss(config.loss, weight=label_weights), config)

    def get_pred(self, logit, *args, **kwargs):
        """Compute and return prediction and scores from the model.

        Prediction is computed using argmax over the document label/target space.

        Scores are sigmoid or softmax scores over the model logits depending on
        the loss component being used.

        Args:
            logit (torch.Tensor): Logits returned
                :class:`~pytext.models.doc_model.DocModel`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Model prediction and scores.

        """
        raise NotImplementedError


class ClassificationScores(jit.ScriptModule):
    def __init__(self, classes, score_function):
        super().__init__()
        self.classes = jit.Attribute(classes, List[str])
        self.score_function = score_function

    @jit.script_method
    def forward(self, logits: torch.Tensor):
        # In pure python, this code would be implemented as follows:
        #   scores = self.score_function(logits)
        #   return [
        #     {class: score for class, score in zip(self.classes, example_scores}
        #     for example_scores in scores.tolist()
        #   ]
        # Extra verbosity is due to jit.script.
        scores = self.score_function(logits)
        results = jit.annotate(List[Dict[str, float]], [])
        for example_scores in scores.chunk(len(scores)):
            example_scores = example_scores.squeeze(dim=0)
            example_response = jit.annotate(Dict[str, float], {})
            for i in range(len(self.classes)):
                example_response[self.classes[i]] = example_scores[i].item()
            results.append(example_response)
        return results


class BinaryClassificationOutputLayer(ClassificationOutputLayer):
    def get_pred(self, logit, *args, **kwargs):
        """See `OutputLayerBase.get_pred()`."""
        preds = torch.max(logit, 1)[1]
        scores = F.logsigmoid(logit)
        return preds, scores

    def torchscript_predictions(self):
        return ClassificationScores(self.target_names, F.logsigmoid)

    def export_to_caffe2(
        self,
        workspace: core.workspace,
        init_net: core.Net,
        predict_net: core.Net,
        model_out: torch.Tensor,
        output_name: str,
    ) -> List[core.BlobReference]:
        """See `OutputLayerBase.export_to_caffe2()`."""
        probability_out = predict_net.Sigmoid(output_name)
        return OutputLayerUtils.gen_additional_blobs(
            predict_net, probability_out, model_out, output_name, self.target_names
        )


class MulticlassOutputLayer(ClassificationOutputLayer):
    def get_pred(self, logit, *args, **kwargs):
        """See `OutputLayerBase.get_pred()`."""
        preds = torch.max(logit, 1)[1]
        scores = F.log_softmax(logit, 1)
        return preds, scores

    def torchscript_predictions(self):
        return ClassificationScores(self.target_names, F.log_softmax)

    def export_to_caffe2(
        self,
        workspace: core.workspace,
        init_net: core.Net,
        predict_net: core.Net,
        model_out: torch.Tensor,
        output_name: str,
    ) -> List[core.BlobReference]:
        """See `OutputLayerBase.export_to_caffe2()`."""
        probability_out = predict_net.Softmax(output_name, axis=model_out.dim() - 1)
        return OutputLayerUtils.gen_additional_blobs(
            predict_net, probability_out, model_out, output_name, self.target_names
        )

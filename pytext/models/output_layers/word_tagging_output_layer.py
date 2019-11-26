#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.jit as jit
import torch.nn as nn
import torch.nn.functional as F
from caffe2.python import core
from pytext.common import Padding
from pytext.config.component import create_loss
from pytext.config.serialize import MissingValueError
from pytext.data.utils import Vocabulary
from pytext.loss import (
    AUCPRHingeLoss,
    BinaryCrossEntropyLoss,
    CrossEntropyLoss,
    KLDivergenceBCELoss,
    KLDivergenceCELoss,
    LabelSmoothedCrossEntropyLoss,
)
from pytext.models.crf import CRF
from pytext.utils.label import get_label_weights

from .output_layer_base import OutputLayerBase
from .utils import OutputLayerUtils


class WordTaggingScores(nn.Module):
    classes: List[str]

    def __init__(self, classes):
        super().__init__()
        self.classes = classes

    def forward(
        self, logits: torch.Tensor, context: Optional[Dict[str, torch.Tensor]] = None
    ) -> List[List[Dict[str, float]]]:
        scores: torch.Tensor = F.log_softmax(logits, 2)
        return _get_prediction_from_scores(scores, self.classes)


class CRFWordTaggingScores(WordTaggingScores):
    def __init__(self, classes: List[str], crf):
        super().__init__(classes)
        self.crf = crf
        self.crf.eval()

    def forward(
        self, logits: torch.Tensor, context: Dict[str, torch.Tensor]
    ) -> List[List[Dict[str, float]]]:
        # We need seq_lengths for CRF decode
        assert "seq_lens" in context
        pred = self.crf.decode(logits, context["seq_lens"])
        logits_rearranged = _rearrange_output(logits, pred)
        scores: torch.Tensor = F.log_softmax(logits_rearranged, 2)
        return _get_prediction_from_scores(scores, self.classes)


class WordTaggingOutputLayer(OutputLayerBase):
    """
    Output layer for word tagging models. It supports `CrossEntropyLoss` per word.

    Args:
        loss_fn (CrossEntropyLoss): Cross-entropy loss component. Defaults to None.

    Attributes:
        loss_fn: Cross-entropy loss component.

    """

    class Config(OutputLayerBase.Config):
        loss: Union[
            CrossEntropyLoss.Config,
            BinaryCrossEntropyLoss.Config,
            AUCPRHingeLoss.Config,
            KLDivergenceBCELoss.Config,
            KLDivergenceCELoss.Config,
            LabelSmoothedCrossEntropyLoss.Config,
        ] = CrossEntropyLoss.Config()
        label_weights: Dict[str, float] = {}
        ignore_pad_in_loss: Optional[bool] = True

    @classmethod
    def from_config(cls, config: Config, labels: Vocabulary):
        vocab = list(labels)
        vocab_dict = labels.idx
        pad_token_idx = labels.idx.get(labels.pad_token, Padding.DEFAULT_LABEL_PAD_IDX)
        label_weights = (
            get_label_weights(vocab_dict, config.label_weights)
            if config.label_weights
            else None
        )
        return cls(
            vocab,
            create_loss(
                config.loss,
                weight=label_weights,
                ignore_index=pad_token_idx if config.ignore_pad_in_loss else -1,
            ),
        )

    def get_loss(
        self,
        logit: torch.Tensor,
        target: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        context: Dict[str, Any],
        reduce: bool = True,
    ) -> torch.Tensor:
        """Compute word tagging loss by comparing prediction of each word in the
        sentence with its true label/target.

        Args:
            logit (torch.Tensor): Logit returned by
                :class:`~pytext.models.word_model.WordTaggingModel`.
            targets (torch.Tensor): True document label/target.
            context (Dict[str, Any]): Context is a dictionary of items
                that's passed as additional metadata. Defaults to None.
            reduce (bool): Whether to reduce loss over the batch. Defaults to True.

        Returns:
            torch.Tensor: Word tagging loss for all words in the sentence.

        """
        # flatten the logit from [batch_size, seq_lens, dim] to
        # [batch_size * seq_lens, dim]
        flattened_logit = logit.view(-1, logit.size()[-1])
        if isinstance(target, tuple):
            hard_target, _, soft_target = target
            target = (
                hard_target.view(-1),
                None,
                soft_target.view(-1, soft_target.size()[-1]),
            )
            return self.loss_fn(flattened_logit, target, reduce)

        return self.loss_fn(flattened_logit, target.view(-1), reduce)

    def get_pred(
        self, logit: torch.Tensor, *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute and return prediction and scores from the model.
        Prediction is computed using argmax over the word label/target space.
        Scores are softmax scores over the model logits.

        Args:
            logit (torch.Tensor): Logits returned
                :class:`~pytext.models.word_model.WordTaggingModel`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Model prediction and scores.

        """
        preds = torch.max(logit, 2)[1]
        scores = F.log_softmax(logit, 2)
        return preds, scores

    def export_to_caffe2(
        self,
        workspace: core.workspace,
        init_net: core.Net,
        predict_net: core.Net,
        model_out: torch.Tensor,
        output_name: str,
    ) -> List[core.BlobReference]:
        """Exports the word tagging output layer to Caffe2."""
        probability_out = predict_net.Softmax(output_name, axis=model_out.dim() - 1)
        return OutputLayerUtils.gen_additional_blobs(
            predict_net, probability_out, model_out, output_name, self.target_names
        )

    def torchscript_predictions(self):
        return jit.script(WordTaggingScores(self.target_names))


class CRFOutputLayer(OutputLayerBase):
    """
    Output layer for word tagging models that use Conditional Random Field.

    Args:
        num_tags (int): Total number of possible word tags.

    Attributes:
        num_tags: Total number of possible word tags.

    """

    __EXPANSIBLE__ = True

    @classmethod
    def from_config(cls, config: OutputLayerBase.Config, labels: Vocabulary):
        vocab_size = len(labels)
        return cls(vocab_size, labels)

    def __init__(self, num_tags, labels: Vocabulary, *args) -> None:
        super().__init__(list(labels), *args)
        self.crf = CRF(
            num_tags=num_tags,
            ignore_index=labels.get_pad_index(Padding.DEFAULT_LABEL_PAD_IDX),
            default_label_pad_index=Padding.DEFAULT_LABEL_PAD_IDX,
        )

    def get_loss(
        self,
        logit: torch.Tensor,
        target: torch.Tensor,
        context: Dict[str, Any],
        reduce=True,
    ):
        """Compute word tagging loss by using CRF.

        Args:
            logit (torch.Tensor): Logit returned by
                :class:`~pytext.models.WordTaggingModel`.
            targets (torch.Tensor): True document label/target.
            context (Dict[str, Any]): Context is a dictionary of items
                that's passed as additional metadata. Defaults to None.
            reduce (bool): Whether to reduce loss over the batch. Defaults to True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Model prediction and scores.

        """
        loss = -1 * self.crf(logit, target, reduce=False)
        return loss.mean() if reduce else loss

    def get_pred(
        self,
        logit: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Compute and return prediction and scores from the model.

        Prediction is computed using CRF decoding.

        Scores are softmax scores over the model logits where the logits are
        computed by rearranging the word logits such that decoded word tag has
        the highest valued logits. This is done because with CRF, the highest valued
        word tag for a given may not be part of the overall set of word tags. In
        order for argmax to work, we rearrange the logit values.

        Args:
            logit (torch.Tensor): Logits returned
                :class:`~pytext.models.WordTaggingModel`.
            target (torch.Tensor): Not applicable. Defaults to None.
            context (Optional[Dict[str, Any]]): Context is a dictionary of items
                that's passed as additional metadata. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Model prediction and scores.

        """
        if not context:
            raise MissingValueError("Expected non-None context but got None.")
        pred = self.crf.decode(logit, context["seq_lens"])
        logit_rearranged = _rearrange_output(logit, pred)
        scores = F.log_softmax(logit_rearranged, 2)
        return pred, scores

    def export_to_caffe2(
        self,
        workspace: core.workspace,
        init_net: core.Net,
        predict_net: core.Net,
        model_out: torch.Tensor,
        output_name: str,
    ) -> List[core.BlobReference]:
        """
        Exports the CRF output layer to Caffe2.
        See `OutputLayerBase.export_to_caffe2()` for details.
        """
        output_score = self.crf.export_to_caffe2(
            workspace, init_net, predict_net, output_name
        )
        probability_out = predict_net.Softmax(output_score, axis=model_out.dim() - 1)
        return OutputLayerUtils.gen_additional_blobs(
            predict_net, probability_out, model_out, output_name, self.target_names
        )

    def torchscript_predictions(self):
        return jit.script(CRFWordTaggingScores(self.target_names, jit.script(self.crf)))


@jit.script
def _rearrange_output(logit, pred):
    """
    Rearrange the word logits so that the decoded word has the highest valued
    logits by swapping the indices predicted with those with maximum logits.
    """
    max_logits, max_logit_indices = torch.max(logit, 2, keepdim=True)
    pred_indices = pred.unsqueeze(2)
    pred_logits = torch.gather(logit, 2, pred_indices)
    logit_rearranged = logit.scatter(2, pred_indices, max_logits)
    logit_rearranged.scatter_(2, max_logit_indices, pred_logits)
    return logit_rearranged


@jit.script
def _get_prediction_from_scores(
    scores: torch.Tensor, classes: List[str]
) -> List[List[Dict[str, float]]]:
    """
    Given scores for a batch, get the prediction for each word in the form of a
    List[List[Dict[str, float]]] for callers of the torchscript model to consume.
    The outer list iterates over batches of sentences and the inner iterates
    over each token in the sentence. The dictionary consists of
    `label:score` for each word.

    Example:

    Assuming slot labels are [No-Label, Number, Name]
    Utterances: [[call john please], [Brightness 25]]
    Output could look like:
    [
        [
            { No-Label: -0.1, Number: -1.5, Name: -9.01},
            { No-Label: -2.1, Number: -1.5, Name: -0.01},
            { No-Label: -0.1, Number: -1.5, Name: -2.01},
        ],
        [
            { No-Label: -0.1, Number: -1.5, Name: -9.01},
            { No-Label: -2.1, Number: -0.5, Name: -7.01},
            { No-Label: -0.1, Number: -1.5, Name: -2.01},
        ]
    ]
    """
    results: List[List[Dict[str, float]]] = []
    # Extra verbosity because jit doesn't support zip
    for sentence_scores in scores.chunk(len(scores)):
        sentence_scores = sentence_scores.squeeze(0)
        sentence_response: List[Dict[str, float]] = []
        for word_scores in sentence_scores.chunk(len(sentence_scores)):
            word_scores = word_scores.squeeze(0)
            word_response: Dict[str, float] = {}
            for i in range(len(classes)):
                word_response[classes[i]] = float(word_scores[i].item())
            sentence_response.append(word_response)
        results.append(sentence_response)
    return results

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from caffe2.python import core
from pytext.common import Padding
from pytext.config.component import create_loss
from pytext.config.serialize import MissingValueError
from pytext.data.joint_data_handler import SEQ_LENS  # TODO move to constant
from pytext.data.utils import Vocabulary
from pytext.fields import FieldMeta
from pytext.loss import CrossEntropyLoss
from pytext.models.crf import CRF
from pytext.utils.label import get_label_weights

from .output_layer_base import OutputLayerBase
from .utils import OutputLayerUtils


class WordTaggingOutputLayer(OutputLayerBase):
    """
    Output layer for word tagging models. It supports `CrossEntropyLoss` per word.

    Args:
        loss_fn (CrossEntropyLoss): Cross-entropy loss component. Defaults to None.

    Attributes:
        loss_fn: Cross-entropy loss component.

    """

    class Config(OutputLayerBase.Config):
        loss: CrossEntropyLoss.Config = CrossEntropyLoss.Config()
        label_weights: Dict[str, float] = {}

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
            pad_token_idx = labels.idx.get(
                labels.pad_token, Padding.DEFAULT_LABEL_PAD_IDX
            )
        else:
            vocab = metadata.vocab.itos
            pad_token_idx = metadata.pad_token_idx
            vocab_dict = metadata.vocab.stoi
        label_weights = (
            get_label_weights(vocab_dict, config.label_weights)
            if config.label_weights
            else None
        )
        return cls(
            vocab,
            create_loss(config.loss, weight=label_weights, ignore_index=pad_token_idx),
        )

    def get_loss(
        self,
        logit: torch.Tensor,
        target: torch.Tensor,
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
                that's passed as additional metadata by the
                :class:`~pytext.data.JointModelDataHandler`. Defaults to None.
            reduce (bool): Whether to reduce loss over the batch. Defaults to True.

        Returns:
            torch.Tensor: Word tagging loss for all words in the sentence.

        """
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


class CRFOutputLayer(OutputLayerBase):
    """
    Output layer for word tagging models that use Conditional Random Field.

    Args:
        num_tags (int): Total number of possible word tags.

    Attributes:
        num_tags: Total number of possible word tags.

    """

    @classmethod
    def from_config(
        cls,
        config: OutputLayerBase.Config,
        metadata: Optional[FieldMeta] = None,
        labels: Optional[Vocabulary] = None,
    ):
        if labels is not None:
            vocab = list(labels)
            vocab_size = len(labels)
        else:
            vocab = metadata.vocab.itos
            vocab_size = metadata.vocab_size
        return cls(vocab_size, vocab)

    def __init__(self, num_tags, *args) -> None:
        super().__init__(*args)
        self.crf = CRF(num_tags)

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
                that's passed as additional metadata by the
                :class:`~pytext.data.JointModelDataHandler`. Defaults to None.
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
                that's passed as additional metadata by the
                :class:`~pytext.data.JointModelDataHandler`. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Model prediction and scores.

        """
        if not context:
            raise MissingValueError("Expected non-None context but got None.")
        pred = self.crf.decode(logit, context[SEQ_LENS])
        logit = _rearrange_output(logit, pred)
        return pred, F.log_softmax(logit, 2)

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


# TODO T33442979 remove this after exposing prediction in caffe2 model
def _rearrange_output(logit, pred):
    """
    Rearrange the word logits so that the decoded word has the highest valued
    logits.
    """

    """Context: without this line, the logit gets modified in place by this function
    (which appears to be unintentional), causing errors during backpropagation.

    In the old trainer, this didn't matter, since get_preds (and therefore this
    function) runs after backprop, but in the new trainer, get_preds is run
    before backprop because it is bundled with get_loss in train_batch."""
    logit = logit.detach().clone()

    for batch_idx, v_path in enumerate(pred):
        for w_idx, word in enumerate(v_path):
            w_logits = logit[batch_idx][w_idx]
            v_label = word.item()
            # make the word on the optimal path the greatest
            _, maxIndex = torch.max(w_logits, 0)
            w_logits[v_label], w_logits[maxIndex] = (
                w_logits[maxIndex].item(),
                w_logits[v_label].item(),
            )
            logit[batch_idx][w_idx] = w_logits
    return logit

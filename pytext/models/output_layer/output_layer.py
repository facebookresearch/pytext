#!/usr/bin/env python3

from typing import List, Union

import torch
import torch.nn.functional as F
from caffe2.python import core
from pytext.config.component import create_loss

# TODO move to constant
from pytext.data.joint_data_handler import SEQ_LENS
from pytext.fields import FieldMeta
from pytext.loss import AUCPRHingeLoss, BinaryCrossEntropyLoss, CrossEntropyLoss
from pytext.models.crf import CRF
from pytext.models.module import Module
from pytext.utils.cuda_utils import FloatTensor


class OutputLayerBase(Module):
    @classmethod
    def from_config(cls, config, meta: FieldMeta):
        return cls(create_loss(config.loss), config)

    def __init__(self, loss_fn=None, config=None):
        super().__init__(config)
        self.loss_fn = loss_fn

    def get_loss(self, logit, target, context=None, reduce=True):
        return self.loss_fn(logit, target, reduce)

    def get_pred(self, logit, targets=None, context=None):
        return logit, None

    def export_to_caffe2(
        self,
        workspace: core.workspace,
        init_net: core.Net,
        predict_net: core.Net,
        model_out: torch.Tensor,
        output_name: str,
        label_names: List[str],
    ) -> List[core.BlobReference]:
        """
        Exports the output layer to caffe2 by manually adding the necessary
        operators to the init_net and predict net, and returns the list of
        external output blobs to be added to the model. By default this does
        nothing, so sub-classes should implement and override this method if
        necessary.
        """
        return []


class ClassificationOutputLayer(OutputLayerBase):
    @classmethod
    def from_config(cls, config, meta: FieldMeta):
        label_weights = getattr(meta, "label_weights", None)
        if label_weights is not None:
            label_weights = FloatTensor(label_weights)
        return cls(create_loss(config.loss, weight=label_weights), config)

    class Config(OutputLayerBase.Config):  # noqa: T484
        loss: Union[
            CrossEntropyLoss.Config,
            BinaryCrossEntropyLoss.Config,
            AUCPRHingeLoss.Config,
        ] = CrossEntropyLoss.Config()

    def get_pred(self, logit, targets, context):
        preds = torch.max(logit, 1)[1]
        # Hacky way to check loss type
        if isinstance(self.loss_fn, BinaryCrossEntropyLoss):
            scores = F.logsigmoid(logit)
        else:
            scores = F.log_softmax(logit, 1)
        return preds, scores

    def export_to_caffe2(
        self,
        workspace: core.workspace,
        init_net: core.Net,
        predict_net: core.Net,
        model_out: torch.Tensor,
        output_name: str,
        label_names: List[str],
    ) -> List[core.BlobReference]:
        """
        Exports the doc classification layer to caffe2 by manually adding the
        necessary operators to the init_net and predict net, and returns the
        list of external output blobs to be added to the model.
        """
        if isinstance(self.loss_fn, BinaryCrossEntropyLoss):
            probability_out = predict_net.Sigmoid(output_name)
        else:
            probability_out = predict_net.Softmax(output_name, axis=model_out.dim() - 1)

        return gen_additional_blobs(
            predict_net, probability_out, model_out, output_name, label_names
        )


class CRFOutputLayer(OutputLayerBase):
    @classmethod
    def from_config(cls, config, meta: FieldMeta):
        return cls(meta.vocab_size)

    def __init__(self, num_tags):
        super().__init__()
        self.crf = CRF(num_tags)

    def get_loss(self, logit, target, context, reduce=True):
        loss = -1 * self.crf(logit, target, reduce=False)
        return loss.mean() if reduce else loss

    def get_pred(self, logit, target, context):
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
        label_names: List[str],
    ) -> List[core.BlobReference]:
        """
        Exports the CRF output layer to caffe2 by manually adding the necessary
        operators to the init_net and predict net, and returns the list of
        external output blobs to be added to the model.
        """
        output_score = self.crf.export_to_caffe2(
            workspace, init_net, predict_net, output_name
        )
        probability_out = predict_net.Softmax(output_score, axis=model_out.dim() - 1)
        return gen_additional_blobs(
            predict_net, probability_out, model_out, output_name, label_names
        )


# TODO T33442979 remove this after exposing prediction in caffe2 model
def _rearrange_output(logit, pred):
    """
    rearrange the word logits so that the decoded word has the highest logits
    """
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


def gen_additional_blobs(
    predict_net: core.Net,
    probability_out,
    model_out: torch.Tensor,
    output_name: str,
    label_names: List[str],
) -> List[core.BlobReference]:
    """
    Utility method to generate additional blobs for human readable result.
    """
    res = []
    tmp_out_score = predict_net.Log(probability_out)
    label_scores = predict_net.Split(
        tmp_out_score, label_names, axis=model_out.dim() - 1
    )

    # Make sure label_scores is iterable
    if not isinstance(label_scores, tuple):
        label_scores = (label_scores,)
    for name, label_score in zip(label_names, label_scores):
        res.append(predict_net.Copy(label_score, "{}:{}".format(output_name, name)))
    return res

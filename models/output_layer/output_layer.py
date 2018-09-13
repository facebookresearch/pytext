#!/usr/bin/env python3

from typing import Union

import torch
import torch.nn.functional as F
from pytext.common.constants import DatasetFieldName
from pytext.config import ConfigBase
from pytext.config.component import create_loss
from pytext.data import CommonMetadata

# TODO move to constant
from pytext.data.joint_data_handler import SEQ_LENS
from pytext.loss import BinaryCrossEntropyLoss, CrossEntropyLoss
from pytext.models.crf import CRF
from pytext.models.module import Module


class OutputLayerBase(Module):
    @classmethod
    def from_config(cls, config, meta: CommonMetadata):
        return cls(create_loss(config.loss), config)

    def __init__(self, loss_fn=None, config=None):
        super().__init__(config)
        self.loss_fn = loss_fn

    def get_loss(self, logit, target, context=None, reduce=True):
        return self.loss_fn(logit, target, reduce)

    def get_pred(self, logit, context=None):
        return logit, None


class ClassificationOutputLayer(OutputLayerBase):
    class Config(ConfigBase):
        loss: Union[
            CrossEntropyLoss.Config,
            BinaryCrossEntropyLoss.Config,
        ] = CrossEntropyLoss.Config()

    def get_pred(self, logit, context):
        preds = torch.max(logit, 1)[1]
        # Hacky way to check loss type
        if isinstance(self.loss_fn, BinaryCrossEntropyLoss):
            scores = F.logsigmoid(logit)
        else:
            scores = F.log_softmax(logit, 1)
        return preds, scores


class CRFOutputLayer(OutputLayerBase):
    @classmethod
    def from_config(cls, config, meta: CommonMetadata):
        label_meta = meta.labels[DatasetFieldName.WORD_LABEL_FIELD]
        return cls(label_meta.vocab_size)

    def __init__(self, num_tags):
        super().__init__()
        self.crf = CRF(num_tags)

    def get_loss(self, logit, target, context, reduce=True):
        loss = -1 * self.crf(
            torch.transpose(logit, 0, 1).contiguous(),
            torch.transpose(target, 0, 1).contiguous(),
            reduce=False,
        )
        return loss.mean() if reduce else loss

    def get_pred(self, logit, context):
        pred = self.crf.decode(logit, context[SEQ_LENS])
        logit = _rearrange_output(logit, pred)
        return pred, F.log_softmax(logit, 2)


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

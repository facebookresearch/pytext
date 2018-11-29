#!/usr/bin/env python3
import torch
import torch.nn.functional as F
from pytext.common.constants import DatasetFieldName
from pytext.config.component import create_loss
from pytext.fields import FieldMeta
from pytext.loss import CrossEntropyLoss

from .output_layer import OutputLayerBase


class LMOutputLayer(OutputLayerBase):
    class Config(OutputLayerBase.Config):
        loss: CrossEntropyLoss.Config = CrossEntropyLoss.Config()

    @classmethod
    def from_config(cls, config, meta: FieldMeta):
        return cls(
            meta.vocab.itos,
            create_loss(config.loss, ignore_index=meta.pad_token_idx),
            pad_token_idx=meta.pad_token_idx,
        )

    def __init__(self, target_names, loss_fn=None, config=None, pad_token_idx=-100):
        super().__init__(target_names, loss_fn, config)
        self.pad_token_idx = pad_token_idx

    def get_pred(self, logit, target, context):
        # Shape of logit: (bsize x seq_len x vocab)
        # Reshape m_out to (bsize x vocab x seq_len) for cross_entropy_loss
        logit = logit.transpose(1, 2)
        # loss dim: (bsize x seq_len)
        loss = F.cross_entropy(
            logit, target, reduce=False, ignore_index=self.pad_token_idx
        )
        # context[DatasetFieldName.SEQ_LENS] s the length of each sequence
        # sequence_loss is the loss per word for each sequence in the batch
        # sequence_loss dim: (bsize,)
        sequence_loss = loss.sum(1) / context[DatasetFieldName.TARGET_SEQ_LENS].float()
        scores = self.calculate_perplexity(sequence_loss)
        return scores, scores

    def get_loss(self, logit, target, context, reduce=True):
        # flatten the logit from [batch_size, seq_lens, dim] to
        # [batch_size * seq_lens, dim]
        return self.loss_fn(logit.view(-1, logit.size()[-1]), target.view(-1), reduce)

    @staticmethod
    def calculate_perplexity(sequence_loss: torch.Tensor) -> torch.Tensor:
        return torch.exp(sequence_loss)

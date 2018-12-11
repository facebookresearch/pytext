#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import List, Union

import torch
import torch.nn.functional as F
from caffe2.python import core
from pytext.config.component import create_loss
from pytext.fields import FieldMeta
from pytext.loss import AUCPRHingeLoss, BinaryCrossEntropyLoss, CrossEntropyLoss
from pytext.utils.cuda_utils import FloatTensor

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
        ] = CrossEntropyLoss.Config()

    @classmethod
    def from_config(cls, config: Config, metadata: FieldMeta):
        label_weights = getattr(metadata, "label_weights", None)
        if label_weights is not None:
            label_weights = FloatTensor(label_weights)
        return cls(
            metadata.vocab.itos, create_loss(config.loss, weight=label_weights), config
        )

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
    ) -> List[core.BlobReference]:
        """
        Exports the doc classification layer to Caffe2.
        See `OutputLayerBase.export_to_caffe2()` for details.
        """
        if isinstance(self.loss_fn, BinaryCrossEntropyLoss):
            probability_out = predict_net.Sigmoid(output_name)
        else:
            probability_out = predict_net.Softmax(output_name, axis=model_out.dim() - 1)

        return OutputLayerUtils.gen_additional_blobs(
            predict_net, probability_out, model_out, output_name, self.target_names
        )

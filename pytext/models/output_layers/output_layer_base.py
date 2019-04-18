#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Any, Dict, List, Optional, Tuple

import torch
from caffe2.python import core
from pytext.config.component import create_loss
from pytext.fields import FieldMeta
from pytext.loss import Loss
from pytext.models.module import Module


class OutputLayerBase(Module):
    """
    Base class for all output layers in PyText. The responsibilities of this layer are

        1. Implement how loss is computed from logits and targets.
        2. Implement how to get predictions from logits.
        3. Implement the Caffe2 operator for performing the above tasks. This is
            used when PyText exports PyTorch model to Caffe2.

    Args:
        loss_fn (type): The loss function object to use for computing loss.
            Defaults to None.

    Attributes:
        loss_fn: The loss function object to use for computing loss.

    """

    def __init__(
        self,
        target_names: Optional[List[str]] = None,
        loss_fn: Optional[Loss] = None,
        *args,
        **kwargs
    ) -> None:
        super().__init__()
        self.target_names = target_names
        self.loss_fn = loss_fn

    def get_loss(
        self,
        logit: torch.Tensor,
        target: torch.Tensor,
        context: Optional[Dict[str, Any]] = None,
        reduce: bool = True,
    ) -> torch.Tensor:
        """Compute and return the loss given logits and targets.

        Args:
            logit (torch.Tensor): Logits returned :class:`~pytext.models.Model`.
            target (torch.Tensor): True label/target to compute loss against.
            context (Optional[Dict[str, Any]]): Context is a dictionary of items
                that's passed as additional metadata by the
                :class:`~pytext.data.DataHandler`. Defaults to None.
            reduce (bool): Whether to reduce loss over the batch. Defaults to True.

        Returns:
            torch.Tensor: Model loss.

        """
        return self.loss_fn(logit, target, reduce) if self.loss_fn else None

    def get_pred(
        self,
        logit: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute and return prediction and scores from the model.

        Args:
            logit (torch.Tensor): Logits returned :class:`~pytext.models.Model`.
            targets (Optional[torch.Tensor]): True label/target. Only used by
                :class:`~pytext.models.output_layer.LMOutputLayer`. Defaults to None.
            context (Optional[Dict[str, Any]]): Context is a dictionary of items
                that's passed as additional metadata by the
                :class:`~pytext.data.DataHandler`. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Model prediction and scores.

        """
        return logit, None

    def export_to_caffe2(
        self,
        workspace: core.workspace,
        init_net: core.Net,
        predict_net: core.Net,
        model_out: torch.Tensor,
        output_name: str,
    ) -> List[core.BlobReference]:
        """
        Exports the output layer to Caffe2 by manually adding the necessary operators
        to the init_net and predict_net and, returns the list of external output
        blobs to be added to the model. By default this does nothing, so any
        sub-class must override this method (if necessary).

        To learn about Caffe2 computation graphs and why we need two networks,
        `init_net` and `predict_net`/`exec_net` read
        https://caffe2.ai/docs/intro-tutorial#null__nets-and-operators.

        Args:
            workspace (core.workspace): Caffe2 `workspace` to use for adding the
                operator. See https://caffe2.ai/docs/workspace.html to learn about
                Caffe2 workspace.
            init_net (core.Net): Caffe2 `init_net` to add the operator to.
            predict_net (core.Net): Caffe2 `predict_net` to add the operator to.
            model_out (torch.Tensor): Output logit Tensor from the model to .
            output_name (str): Name of `model_out` to use in Caffe2 net.
            label_names (List[str]): List of names of the targets/labels to
                expose from the Caffe2 net.

        Returns:
            List[core.BlobReference]: List of output blobs that the `output_layer`
                generates.

        """
        return []

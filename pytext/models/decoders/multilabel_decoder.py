#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Dict, List

import torch
import torch.nn as nn
from pytext.utils.usage import log_class_usage

from .decoder_base import DecoderBase


class MultiLabelDecoder(DecoderBase):
    """
    Implements a 'n-tower' MLP: one for each of the multi labels
    Used in USM/EA: the user satisfaction modeling, pTSR prediction and
    Error Attribution are all 3 label sets that need predicting.

    """

    class Config(DecoderBase.Config):
        # Intermediate hidden dimensions
        hidden_dims: List[int] = []

    def __init__(
        self,
        config: Config,
        in_dim: int,
        output_dim: Dict[str, int],
        label_names: List[str],
    ) -> None:
        super().__init__(config)
        self.label_mlps = nn.ModuleDict({})
        # Store the ordered list to preserve the ordering of the labels
        # when generating the output layer
        self.label_names = label_names
        aggregate_out_dim = 0
        for label_, _ in output_dim.items():
            self.label_mlps[label_] = MultiLabelDecoder.get_mlp(
                in_dim, output_dim[label_], config.hidden_dims
            )
            aggregate_out_dim += output_dim[label_]
        self.out_dim = (1, aggregate_out_dim)
        log_class_usage(__class__)

    @staticmethod
    def get_mlp(in_dim: int, out_dim: int, hidden_dims: List[int]):
        layers = []
        current_dim = in_dim
        for dim in hidden_dims or []:
            layers.append(nn.Linear(current_dim, dim))
            layers.append(nn.ReLU())
            current_dim = dim
        layers.append(nn.Linear(current_dim, out_dim))
        return nn.Sequential(*layers)

    def forward(self, *input: torch.Tensor):
        logits = tuple(
            self.label_mlps[x](torch.cat(input, 1)) for x in self.label_names
        )
        return logits

    def get_decoder(self) -> List[nn.Module]:
        return self.label_mlps

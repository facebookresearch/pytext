#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytext.config.module_config import CNNParams
from pytext.utils import lazy

from .representation_base import RepresentationBase


class DocNNRepresentation(RepresentationBase):
    """CNN based representation of a document."""

    class Config(RepresentationBase.Config):
        dropout: float = 0.4
        cnn: CNNParams = CNNParams()

    @classmethod
    def from_config(cls, config: Config, embed_dim: int = None):
        # Lazy so we don't need the embed_dim
        return cls(config.cnn.kernel_num, config.cnn.kernel_sizes, config.dropout)

    def __init__(
        self,
        kernel_num: int = 100,
        kernel_sizes: List[int] = (3, 4),
        dropout: float = 0.4,
    ) -> None:
        """
        Args:
            kernel_num: Number of feature maps for each Conv1d kernel
            kernel_sizes: A Conv1d kernel will be created with the corresponding size
                for each kernel in this list; the results will be concatenated together
                to form the final representation
            dropout: Dropout ratio applied to the final representation during training
        """
        nn.Module.__init__(self)
        self.max_kernel = max(kernel_sizes)
        self.convs = nn.ModuleList(
            [lazy.Conv1d(kernel_num, K, padding=K) for K in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)

        self.representation_dim = len(kernel_sizes) * kernel_num

    def forward(self, embedded_tokens: torch.Tensor, *args) -> torch.Tensor:
        """Accepts a token embedding of size
            (batch, sequence_length, embedding)
        and returns a representation of size
            (batch, (len(kernel_sizes) * kernel_num))
        """
        # embedded_tokens of size (N,W,D)
        rep = embedded_tokens
        # nn.Conv1d expects a tensor of dim (batch_size x embed_dim x seq_len)
        rep = rep.transpose(1, 2)
        rep = [self.conv_and_pool(rep, conv) for conv in self.convs]
        rep = self.dropout(torch.cat(rep, 1))  # (N,len(Ks)*Co)
        return rep

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x))
        x, _ = torch.max(x, dim=2)
        return x

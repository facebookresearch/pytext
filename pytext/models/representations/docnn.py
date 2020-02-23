#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytext.config import ConfigBase
from pytext.config.module_config import CNNParams, PoolingType
from pytext.utils.usage import log_class_usage

from .representation_base import RepresentationBase


class DocNNRepresentation(RepresentationBase):
    """CNN based representation of a document."""

    class Config(RepresentationBase.Config):
        dropout: float = 0.4
        cnn: CNNParams = CNNParams()
        pooling: PoolingType = PoolingType.MAX

    def __init__(self, config: Config, embed_dim: int) -> None:
        super().__init__(config)
        self.max_kernel = max(config.cnn.kernel_sizes)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(embed_dim, config.cnn.kernel_num, K, padding=K)
                for K in config.cnn.kernel_sizes
            ]
        )
        self.dropout = nn.Dropout(config.dropout)
        self.representation_dim = len(config.cnn.kernel_sizes) * config.cnn.kernel_num
        self.pooling_type = config.pooling
        log_class_usage(__class__)

    def forward(self, embedded_tokens: torch.Tensor, *args) -> torch.Tensor:
        # embedded_tokens of size (N,W,D)
        rep = embedded_tokens
        # nn.Conv1d expects a tensor of dim (batch_size x embed_dim x seq_len)
        rep = rep.transpose(1, 2)
        rep = [self.conv_and_pool(rep, conv) for conv in self.convs]
        rep = self.dropout(torch.cat(rep, 1))  # (N,len(Ks)*Co)
        return rep

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x))
        if self.pooling_type == PoolingType.MAX:
            x, _ = torch.max(x, dim=2)
        elif self.pooling_type == PoolingType.MEAN:
            x = torch.mean(x, dim=2)
        elif self.pooling_type == PoolingType.LOGSUMEXP:
            x = torch.logsumexp(x, dim=2)
        else:
            raise NotImplementedError
        return x

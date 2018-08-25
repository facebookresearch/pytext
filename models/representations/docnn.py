#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytext.config import ConfigBase
from pytext.config.module_config import CNNParams

from .representation_base import RepresentationBase


class DocNNRepresentation(RepresentationBase):
    """CNN based representation of a document."""

    class Config(ConfigBase):
        dropout: float = 0.4
        cnn: CNNParams = CNNParams()

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

    def forward(self, tokens: torch.Tensor, *args) -> torch.Tensor:
        # tokens of size (N,W,D)
        rep = tokens
        # Turn (batch_size * seq_len x input_size) into
        # (batch_size x input_size x seq_len) for CNN
        rep = rep.transpose(1, 2)
        rep = [self.conv_and_pool(rep, conv) for conv in self.convs]
        rep = self.dropout(torch.cat(rep, 1))  # (N,len(Ks)*Co)
        return rep

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x))
        x, _ = torch.max(x, dim=2)
        return x

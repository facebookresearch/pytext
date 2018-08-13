#!/usr/bin/env python3

import torch
import torch.nn as nn
from typing import List
import torch.nn.functional as F
from .representation_base import RepresentationBase


class DocNNRepresentation(RepresentationBase):
    """CNN based representation of a document."""

    def __init__(
        self,
        embedding_dim: int,
        kernel_num: int,
        kernel_sizes: List[int],
        dropout: float,
        pad_idx: int,
    ) -> None:
        super().__init__()
        self.max_kernel = max(kernel_sizes)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embedding_dim, kernel_num, K, padding=K) for K in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.representation_dim = len(kernel_sizes) * kernel_num
        self.pad_idx = pad_idx

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

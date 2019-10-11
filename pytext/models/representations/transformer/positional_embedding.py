#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Optional

import torch
from torch import nn


def make_positions(tensor, pad_index: int):
    """Replace non-padding symbols with their position numbers.
    Position numbers begin at pad_index+1. Padding symbols are ignored.
    """
    masked = tensor.ne(pad_index).long()
    return torch.cumsum(masked, dim=1) * masked + pad_index


class PositionalEmbedding(nn.Module):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on pad_index
    or by setting pad_index to None and ensuring that the appropriate
    position ids are passed to the forward function.

    This is a TorchScriptable implementation of PositionalEmbedding from fairseq
    for the purposes of creating a productionized RoBERTa model. It distills just
    the elements which are required to implement the RoBERTa use cases of
    MultiheadAttention, and within that is restructured and rewritten to be able
    to be compiled by TorchScript for production use cases.
    """

    def __init__(
        self, num_embeddings: int, embedding_dim: int, pad_index: Optional[int] = None
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, pad_index)
        self.pad_index = pad_index

    def forward(self, input):
        """Input is expected to be of size [batch_size x sequence_length]."""
        positions = make_positions(input, self.pad_index)
        return self.embedding(positions)

    def max_positions(self):
        """Maximum number of supported positions."""
        if self.pad_index is not None:
            return self.num_embeddings - self.pad_index - 1
        else:
            return self.num_embeddings

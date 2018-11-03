#!/usr/bin/env python3

from typing import Iterable

import torch
from torch.nn import ModuleList

from .embedding_base import EmbeddingBase


class EmbeddingList(EmbeddingBase, ModuleList):
    """
    Generate a list of sub-embeddings, concat embedding tensors to a single tensor
    or return a tuple of tenors
    Attributes:
        concat: if the embedding tensors should be concatenated
        embeddings: tuple of EmbeddingBase class
        num_emb: how many flattened embeddings in this list,
            e.g: ((e1, e2), e3) has 3 in total
        embedding_dim: total embedding dimension, can be a single int or tuple of
            int depending on concat setting
    """

    def __init__(self, embeddings: Iterable[EmbeddingBase], concat: bool) -> None:
        EmbeddingBase.__init__(self, 0)
        ModuleList.__init__(self, filter(None, embeddings))
        self.concat = concat
        assert len(self) > 0, "must have at least 1 sub embedding"
        self.num_emb = sum(emb.num_emb for emb in self)
        embedding_dims = tuple(emb.embedding_dim for emb in self)
        self.embedding_dim = sum(embedding_dims) if concat else embedding_dims

    def forward(self, *emb_input) -> torch.Tensor:
        """input should match the size of configed embeddings, each is
        either a tensor or a tuple of tensor
        """
        # tokens dim: (bsz, max_seq_len) -> (bsz, max_seq_len, dim)
        # or (bsz, max_num_sen, max_seq_len) -> (bsz, max_num_sen, max_seq_len, dim)
        # for seqnn
        if self.num_emb != len(emb_input):
            raise Exception(
                f"expecting {self.num_emb} embeddings, but get {len(emb_input)} input"
            )
        tensors = []
        start = 0
        for emb in self:
            end = start + emb.num_emb
            input = emb_input[start:end]
            start = end
            # single embedding
            if len(input) == 1:
                # the input for the single embedding is a tuple or list of tensors
                if isinstance(input[0], list) or isinstance(input[0], tuple):
                    [input] = input
            emb_tensor = emb(*input)
            tensors.append(emb_tensor)

        return torch.cat(tensors, 2) if self.concat else tuple(tensors)

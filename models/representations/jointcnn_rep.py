#!/usr/bin/env python3

from typing import List

import torch

from .biseqcnn import BSeqCNNRepresentation
from .docnn import DocNNRepresentation
from .representation_base import RepresentationBase


class JointCNNRepresentation(RepresentationBase):
    def __init__(
        self,
        fwd_bwd_ctxt_len: int,
        surr_ctxt_len: int,
        embedding_dim: int,
        in_channels: int,
        out_channels: int,
        kernel_sizes: int,
        dropout_ratio: float,
        pad_idx: int,
    ) -> None:
        super().__init__()

        ctxt_pad_len = max(fwd_bwd_ctxt_len, surr_ctxt_len)
        self.doc_rep = DocNNRepresentation(
            embedding_dim, out_channels, kernel_sizes, dropout_ratio, pad_idx
        )
        self.word_rep = BSeqCNNRepresentation(
            fwd_bwd_ctxt_len,
            surr_ctxt_len,
            ctxt_pad_len,
            pad_idx,
            embedding_dim,
            in_channels,
            out_channels,
            kernel_sizes,
        )

    def forward(self, tokens: torch.Tensor, *args) -> List[torch.Tensor]:
        return [self.doc_rep(tokens), self.word_rep(tokens)]

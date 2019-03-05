#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Union

import torch
from pytext.config import ConfigBase
from pytext.models.module import create_module

from .bilstm_doc_attention import BiLSTMDocAttention
from .docnn import DocNNRepresentation
from .representation_base import RepresentationBase


class SeqRepresentation(RepresentationBase):
    """
    Representation for a sequence of sentences
    Each sentence will be embedded with a DocNN model,
    then all the sentences are embedded with another DocNN/BiLSTM model
    """

    class Config(RepresentationBase.Config):
        doc_representation: DocNNRepresentation.Config = DocNNRepresentation.Config()
        seq_representation: Union[
            BiLSTMDocAttention.Config, DocNNRepresentation.Config
        ] = BiLSTMDocAttention.Config()

    def __init__(self, config: Config, embed_dim: int) -> None:
        super().__init__(config)
        self.doc_rep = create_module(config.doc_representation, embed_dim=embed_dim)
        self.doc_representation_dim = self.doc_rep.representation_dim

        self.seq_rep = create_module(
            config.seq_representation, embed_dim=self.doc_representation_dim
        )
        self.representation_dim = self.seq_rep.representation_dim

    def forward(
        self, embedded_seqs: torch.Tensor, seq_lengths: torch.Tensor, *args
    ) -> torch.Tensor:
        # embedded_seqs: (bsz, max_num_sen, max_seq_len, dim)
        (bsz, max_num_sen, max_seq_len, dim) = torch.onnx.operators.shape_as_tensor(
            embedded_seqs
        )
        rep = self.doc_rep(
            torch.onnx.operators.reshape_from_tensor_shape(
                embedded_seqs,
                torch.cat(
                    ((bsz * max_num_sen).view(1), max_seq_len.view(1), dim.view(1))
                ),
            )
        )
        sentence_reps = torch.onnx.operators.reshape_from_tensor_shape(
            rep,
            torch.cat(
                (
                    bsz.view(1),
                    max_num_sen.view(1),
                    torch.tensor(self.doc_representation_dim).view(1),
                )
            ),
        )
        if isinstance(self.seq_rep, BiLSTMDocAttention):
            return self.seq_rep(embedded_tokens=sentence_reps, seq_lengths=seq_lengths)
        else:
            return self.seq_rep(embedded_tokens=sentence_reps)

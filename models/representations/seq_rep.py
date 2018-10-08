#!/usr/bin/env python3

from typing import Union

import torch
from pytext.config import ConfigBase
from pytext.config.component import create_module
from .docnn import DocNNRepresentation
from .bilstm_doc_attention import BiLSTMDocAttention
from .representation_base import RepresentationBase


class SeqRepresentation(RepresentationBase):
    """
    Representation for a sequence of sentences
    Each sentence will be embedded with a DocNN model,
    then all the sentences are embedded with another DocNN/BiLSTM model
    """
    class Config(ConfigBase):
        doc_representation: \
            DocNNRepresentation.Config = DocNNRepresentation.Config()
        seq_representation: Union[
            BiLSTMDocAttention.Config,
            DocNNRepresentation.Config
        ] = BiLSTMDocAttention.Config()

    def __init__(self, config: Config, embed_dim: int) -> None:
        super().__init__(config)
        self.doc_rep = create_module(
            config.doc_representation, embed_dim=embed_dim)
        self.doc_representation_dim = self.doc_rep.representation_dim

        self.seq_rep = create_module(
            config.seq_representation, embed_dim=self.doc_representation_dim)
        self.representation_dim = self.seq_rep.representation_dim

    def forward(
        self,
        embedded_seqs: torch.Tensor,
        seq_lengths: torch.Tensor,
        *args
    ) -> torch.Tensor:
        # embedded_seqs: (bsz, max_num_sen, max_seq_len, dim)
        (bsz, max_num_sen, max_seq_len, dim) = embedded_seqs.size()
        rep = self.doc_rep(
            embedded_seqs.reshape((bsz * max_num_sen, max_seq_len, dim))
        )
        sentence_reps = rep.reshape((bsz, max_num_sen, self.doc_representation_dim))
        return self.seq_rep(
            embedded_tokens=sentence_reps,
            seq_lengths=seq_lengths
        )

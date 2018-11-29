#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List

import torch
from pytext.config import ConfigBase

from .biseqcnn import BSeqCNNRepresentation
from .docnn import DocNNRepresentation
from .representation_base import RepresentationBase


class JointCNNRepresentation(RepresentationBase):
    class Config(RepresentationBase.Config):
        doc_representation: DocNNRepresentation.Config = DocNNRepresentation.Config()
        word_representation: BSeqCNNRepresentation.Config = BSeqCNNRepresentation.Config()

    def __init__(self, config: Config, embed_dim: int) -> None:
        super().__init__(config)
        self.doc_rep = DocNNRepresentation(config.doc_representation, embed_dim)
        self.word_rep = BSeqCNNRepresentation(config.word_representation, embed_dim)
        self.doc_representation_dim = self.doc_rep.representation_dim
        self.word_representation_dim = self.word_rep.representation_dim

    def forward(self, embedded_tokens: torch.Tensor, *args) -> List[torch.Tensor]:
        return [self.doc_rep(embedded_tokens), self.word_rep(embedded_tokens)]

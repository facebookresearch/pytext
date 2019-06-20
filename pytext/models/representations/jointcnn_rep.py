#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List, Union

import torch
from pytext.models.module import create_module

from .biseqcnn import BSeqCNNRepresentation
from .deepcnn import DeepCNNRepresentation
from .docnn import DocNNRepresentation
from .representation_base import RepresentationBase


class JointCNNRepresentation(RepresentationBase):
    class Config(RepresentationBase.Config):
        doc_representation: DocNNRepresentation.Config = DocNNRepresentation.Config()
        word_representation: Union[
            BSeqCNNRepresentation.Config, DeepCNNRepresentation.Config
        ] = BSeqCNNRepresentation.Config()

    def __init__(self, config: Config, embed_dim: int) -> None:
        super().__init__(config)
        self.doc_rep = create_module(config.doc_representation, embed_dim)
        self.word_rep = create_module(config.word_representation, embed_dim)
        self.doc_representation_dim = self.doc_rep.representation_dim
        self.word_representation_dim = self.word_rep.representation_dim

    def forward(self, embedded_tokens: torch.Tensor, *args) -> List[torch.Tensor]:
        return [self.doc_rep(embedded_tokens), self.word_rep(embedded_tokens)]

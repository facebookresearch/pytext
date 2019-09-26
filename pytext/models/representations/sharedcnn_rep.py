#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List, Union

import torch
from pytext.config.module_config import PoolingType
from pytext.models.module import create_module

from .biseqcnn import BSeqCNNRepresentation
from .deepcnn import DeepCNNRepresentation
from .representation_base import RepresentationBase


def pool(pooling_type, words):
    # input dims: bsz * seq_len * num_filters
    if pooling_type == PoolingType.MEAN:
        return words.mean(dim=1)
    elif pooling_type == PoolingType.MAX:
        return words.max(dim=1)[0]
    elif pooling_type == PoolingType.NONE:
        return words
    else:
        return NotImplementedError


class SharedCNNRepresentation(RepresentationBase):
    class Config(RepresentationBase.Config):
        word_representation: Union[
            BSeqCNNRepresentation.Config, DeepCNNRepresentation.Config
        ] = DeepCNNRepresentation.Config()
        pooling_type: PoolingType = PoolingType.MAX

    def __init__(self, config: Config, embed_dim: int) -> None:
        super().__init__(config)
        self.word_rep = create_module(config.word_representation, embed_dim)
        self.word_representation_dim = self.word_rep.representation_dim
        self.doc_representation_dim = self.word_rep.representation_dim
        self.pooling_type = config.pooling_type

    def forward(self, embedded_tokens: torch.Tensor, *args) -> List[torch.Tensor]:
        return [
            pool(self.pooling_type, self.word_rep(embedded_tokens)),
            self.word_rep(embedded_tokens),
        ]

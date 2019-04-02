#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from pytext.config import ConfigBase
from pytext.models.decoders.mlp_decoder import MLPDecoder
from pytext.models.module import create_module

from .pooling import BoundaryPool, MaxPool, MeanPool, NoPool, SelfAttention
from .representation_base import RepresentationBase


class PureDocAttention(RepresentationBase):
    """pooling (e.g. max pooling or self attention)
        followed by optional MLP"""

    class Config(RepresentationBase.Config):
        dropout: float = 0.4
        pooling: Union[
            SelfAttention.Config,
            MaxPool.Config,
            MeanPool.Config,
            NoPool.Config,
            BoundaryPool.Config,
        ] = SelfAttention.Config()
        mlp_decoder: Optional[MLPDecoder.Config] = None

    def __init__(self, config: Config, embed_dim: int) -> None:
        """embed_dim is the dimension of embedded_tokens
        """
        super().__init__(config)

        self.dropout = nn.Dropout(config.dropout)

        # Document attention.
        self.attention = (
            create_module(config.pooling, n_input=embed_dim)
            if config.pooling is not None
            else None
        )

        # Non-linear projection over attended representation.
        self.dense = None
        if (
            isinstance(config.pooling, BoundaryPool.Config)
            and config.pooling.boundary_type == "firstlast"
        ):
            # the dimension double because of concatenating bos and eos
            self.representation_dim = embed_dim * 2
        else:
            self.representation_dim = embed_dim

        if config.mlp_decoder:
            self.dense = MLPDecoder(config.mlp_decoder, in_dim=embed_dim)
            self.representation_dim = self.dense.out_dim

    def forward(
        self, embedded_tokens: torch.Tensor, seq_lengths: torch.Tensor = None, *args
    ) -> Any:
        rep = self.dropout(embedded_tokens)

        # Attention
        if self.attention:
            rep = self.attention(rep, seq_lengths)

        # Non-linear projection
        if self.dense:
            rep = self.dense(rep)

        return rep

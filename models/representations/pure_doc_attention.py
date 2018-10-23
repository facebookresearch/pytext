#!/usr/bin/env python3
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from pytext.config import ConfigBase
from pytext.config.component import create_module
from pytext.models.decoders.mlp_decoder import MLPDecoder

from .pooling import MaxPool, MeanPool, NoPool, SelfAttention
from .representation_base import RepresentationBase


class PureDocAttention(RepresentationBase):
    """pooling (e.g. max pooling or self attention)
        followed by optional MLP"""

    class Config(RepresentationBase.Config, ConfigBase):
        dropout: float = 0.4
        pooling: Union[
            SelfAttention.Config, MaxPool.Config, MeanPool.Config, NoPool.Config
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
        self.representation_dim = embed_dim
        if config.mlp_decoder:
            self.dense = MLPDecoder(
                config.mlp_decoder, from_dim=embed_dim
            )
            self.representation_dim = self.dense.out_dim

    def forward(
        self,
        embedded_tokens: torch.Tensor,
        seq_lengths: torch.Tensor,
        *args
    ) -> Any:
        rep = self.dropout(embedded_tokens)

        # Attention
        if self.attention:
            rep = self.attention(rep, seq_lengths)

        # Non-linear projection
        if self.dense:
            rep = self.dense(rep)

        return rep

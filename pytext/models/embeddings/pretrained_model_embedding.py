#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
from pytext.config.field_config import PretrainedModelEmbeddingConfig

from .embedding_base import EmbeddingBase


class PretrainedModelEmbedding(EmbeddingBase):
    """Module for providing token embeddings from a pretrained model."""

    Config = PretrainedModelEmbeddingConfig

    @classmethod
    def from_config(cls, config: PretrainedModelEmbeddingConfig, *args, **kwargs):
        return cls(config.embed_dim)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        if embedding.shape[2] != self.embedding_dim:
            raise ValueError(
                f"Expected {self.embedding_dim} as dimension for pretrained "
                + f"model embedding but got {embedding.shape[2]}."
            )
        return embedding

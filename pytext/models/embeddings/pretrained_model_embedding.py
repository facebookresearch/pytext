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
        if embedding.shape[1] % self.embedding_dim != 0:
            raise ValueError(
                f"Input embedding_dim {embedding.shape[1]} is not a"
                + f" multiple of specified embedding_dim {self.embedding_dim}"
            )
        num_tokens = embedding.shape[1] // self.embedding_dim
        unflattened_embedding = embedding.view(-1, num_tokens, self.embedding_dim)
        return unflattened_embedding

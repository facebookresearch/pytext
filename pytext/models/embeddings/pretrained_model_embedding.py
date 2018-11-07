#!/usr/bin/env python3

import torch
from pytext.config.field_config import PretrainedModelEmbeddingConfig
from pytext.fields import FieldMeta

from .embedding_base import EmbeddingBase


class PretrainedModelEmbedding(EmbeddingBase):
    Config = PretrainedModelEmbeddingConfig

    @classmethod
    def from_config(cls, config: PretrainedModelEmbeddingConfig, meta: FieldMeta):
        return cls(config.embed_dim)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        if embedding.shape[2] != self.embedding_dim:
            raise ValueError(
                f"Expected {self.embedding_dim} as dimension for pretrained_model_embedding but {embedding.shape[2]} received"
            )
        return embedding

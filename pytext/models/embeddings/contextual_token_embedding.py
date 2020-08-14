#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Optional

import torch
from pytext.config.field_config import ContextualTokenEmbeddingConfig
from pytext.models.seq_models.base import PlaceholderIdentity
from torch.nn import Linear

from .embedding_base import EmbeddingBase


class ContextualTokenEmbedding(EmbeddingBase):
    """Module for providing token embeddings from a pretrained model."""

    Config = ContextualTokenEmbeddingConfig

    @classmethod
    def from_config(cls, config: ContextualTokenEmbeddingConfig, *args, **kwargs):
        return cls(config.embed_dim, downsample_dim=config.downsample_dim)

    def __init__(self, embed_dim: int, downsample_dim: Optional[int] = None) -> None:
        super().__init__(embed_dim)
        self.input_embed_dim = embed_dim
        if downsample_dim:
            self.proj = Linear(embed_dim, downsample_dim)
            self.embedding_dim = downsample_dim
        else:
            self.proj = PlaceholderIdentity()

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        embedding_shape = torch.onnx.operators.shape_as_tensor(embedding)

        # Since embeddings vector is flattened, verify its shape correctness.
        if embedding_shape[1].item() % self.input_embed_dim != 0:
            raise ValueError(
                f"Input embedding_dim {embedding_shape[1]} is not a"
                + f" multiple of specified embedding_dim {self.input_embed_dim}"
            )

        # Unflatten embedding Tensor from (batch_size, seq_len * embedding_size)
        # to (batch_size, seq_len, embedding_size).
        num_tokens = embedding_shape[1] // self.input_embed_dim
        new_embedding_shape = torch.cat(
            (
                torch.tensor([-1], dtype=torch.long),
                num_tokens.view(1),
                torch.tensor([self.input_embed_dim], dtype=torch.long),
            )
        )
        reshaped_embed = torch.onnx.operators.reshape_from_tensor_shape(
            embedding, new_embedding_shape
        )
        return self.proj(reshaped_embed)

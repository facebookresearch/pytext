#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from pytext.config.field_config import ContextualTokenEmbeddingConfig

from .embedding_base import EmbeddingBase


class ContextualTokenEmbedding(EmbeddingBase):
    """Module for providing token embeddings from a pretrained model."""

    Config = ContextualTokenEmbeddingConfig

    @classmethod
    def from_config(cls, config: ContextualTokenEmbeddingConfig, *args, **kwargs):
        return cls(config.embed_dim)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        embedding_shape = torch.onnx.operators.shape_as_tensor(embedding)

        # Since embeddings vector is flattened, verify its shape correctness.
        if embedding_shape[1].item() % self.embedding_dim != 0:
            raise ValueError(
                f"Input embedding_dim {embedding_shape[1]} is not a"
                + f" multiple of specified embedding_dim {self.embedding_dim}"
            )

        # Unflatten embedding Tensor from (batch_size, seq_len * embedding_size)
        # to (batch_size, seq_len, embedding_size).
        # TODO (T65593460): Predictor forward compatibility issue: change back
        # once D18945316 is available on predictor:
        # num_tokens = embedding_shape[1] // self.embedding_dim
        num_tokens = (embedding_shape[1] / self.embedding_dim).long()
        new_embedding_shape = torch.cat(
            (
                torch.tensor([-1], dtype=torch.long),
                num_tokens.view(1),
                torch.tensor([self.embedding_dim], dtype=torch.long),
            )
        )
        return torch.onnx.operators.reshape_from_tensor_shape(
            embedding, new_embedding_shape
        )

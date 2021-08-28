#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytext.config.field_config import CharFeatConfig
from pytext.fields import FieldMeta
from pytext.utils.usage import log_class_usage
from pytorch.text.fb.nn.modules.cnn_char_embedding import CNNCharacterEmbedding

from .embedding_base import EmbeddingBase


class CharacterEmbedding(EmbeddingBase):
    """
    Wrapper for character aware CNN embeddings for tokens.
    Attributes:
        embedding (CNNCharacterEmbedding): Character embedding
    """

    Config = CharFeatConfig

    @classmethod
    def from_config(
        cls,
        config: CharFeatConfig,
        metadata: Optional[FieldMeta] = None,
        vocab_size: Optional[int] = None,
    ):
        """Factory method to construct an instance of CharacterEmbedding from
        the module's config object and the field's metadata object.

        Args:
            config (CharFeatConfig): Configuration object specifying all the
                parameters of CharacterEmbedding.
            metadata (FieldMeta): Object containing this field's metadata.

        Returns:
            type: An instance of CharacterEmbedding.

        """
        if vocab_size is None:
            vocab_size = metadata.vocab_size

        return cls(
            vocab_size,
            config.embed_dim,
            config.cnn.kernel_num,
            config.cnn.kernel_sizes,
            config.highway_layers,
            config.projection_dim,
        )

    def __init__(
        self,
        num_embeddings: int,
        embed_dim: int,
        out_channels: int,
        kernel_sizes: List[int],
        highway_layers: int,
        projection_dim: Optional[int],
        *args,
        **kwargs,
    ) -> None:
        output_dim = CNNCharacterEmbedding.output_dim(
            num_kernels=len(kernel_sizes),
            out_channels=out_channels,
            projection_dim=projection_dim,
        )
        super().__init__(output_dim)
        self.embedding = CNNCharacterEmbedding(
            num_embeddings=num_embeddings,
            embed_dim=embed_dim,
            out_channels=out_channels,
            kernel_sizes=kernel_sizes,
            highway_layers=highway_layers,
            projection_dim=projection_dim,
        )
        log_class_usage(__class__)

    def forward(self, chars: torch.Tensor) -> torch.Tensor:
        """See CNNCharacterEmbedding.forward() for details"""
        return self.embedding(chars)

    def load_state_dict(self, loaded_module: nn.Module) -> None:
        """Add backward compatibility to load CharEmbedding from older versions
        In older versions, state dict keys are like "char_embed.weight", "convs.0.weight", etc
        In newer versions, state dict keys are like "embedding.char_embed.weight", "embedding.convs.0.weight"
        """
        try:
            # assume loaded_module was new version, with weights like embedding.convs.0.weight
            super().load_state_dict(loaded_module)
        except RuntimeError:
            # if failed, perhaps loaded_module was old version, with weights like convs.0.weight
            self.embedding.load_state_dict(loaded_module)

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import List, Optional

import torch
from pytext.config.field_config import MLPFeatConfig
from pytext.data.tensorizers import Tensorizer
from pytext.fields import FieldMeta
from pytext.models.embeddings.embedding_base import EmbeddingBase
from pytext.utils.usage import log_class_usage
from torch import nn
from torch.utils.tensorboard import SummaryWriter


class MLPEmbedding(EmbeddingBase):
    """
    An MLP embedding wrapper module around `torch.nn.Embedding` to add
    transformations for float tensors.

    Args:
        num_embeddings (int): Total number of words/tokens (vocabulary size).
        embedding_dim (int): Size of embedding vector.
        embeddings_weight (torch.Tensor): Pretrained weights to initialize the
            embedding table with.
        init_range (List[int]): Range of uniform distribution to initialize the
            weights with if `embeddings_weight` is None.
        mlp_layer_dims (List[int]): List of layer dimensions (if any) to add
            on top of the embedding lookup.

    """

    Config = MLPFeatConfig

    @classmethod
    def from_config(
        cls,
        config: MLPFeatConfig,
        metadata: Optional[FieldMeta] = None,
        tensorizer: Optional[Tensorizer] = None,
        init_from_saved_state: Optional[bool] = False,
    ):
        """Factory method to construct an instance of MLPEmbedding from
        the module's config object and the field's metadata object.

        Args:
            config (MLPFeatConfig): Configuration object specifying all the
            parameters of MLPEmbedding.
            metadata (FieldMeta): Object containing this field's metadata.

        Returns:
            type: An instance of MLPEmbedding.

        """
        if tensorizer is not None:
            embeddings_weight = None
        else:  # This else condition should go away after metadata goes away.
            embeddings_weight = metadata.pretrained_embeds_weight

        return cls(
            embedding_dim=config.embed_dim,
            embeddings_weight=embeddings_weight,
            init_range=config.embedding_init_range,
            init_std=config.embeddding_init_std,
            mlp_layer_dims=config.mlp_layer_dims,
        )

    def __init__(
        self,
        embedding_dim: int = 300,
        embeddings_weight: Optional[torch.Tensor] = None,
        init_range: Optional[List[int]] = None,
        init_std: Optional[float] = None,
        mlp_layer_dims: List[int] = (),
    ) -> None:
        output_embedding_dim = mlp_layer_dims[-1] if mlp_layer_dims else embedding_dim
        EmbeddingBase.__init__(self, embedding_dim=output_embedding_dim)

        # Create MLP layers
        if mlp_layer_dims is None:
            mlp_layer_dims = []
        self.mlp = nn.Sequential(
            *(
                nn.Sequential(nn.Linear(m, n), nn.ReLU())
                for m, n in zip([embedding_dim] + list(mlp_layer_dims), mlp_layer_dims)
            )
        )
        log_class_usage(__class__)

    def forward(self, input):
        return self.mlp(input)

    def visualize(self, summary_writer: SummaryWriter):
        pass

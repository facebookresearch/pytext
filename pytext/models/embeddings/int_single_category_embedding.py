#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Dict, List

import torch
import torch.nn as nn
from pytext.config.module_config import ModuleConfig
from pytext.utils.usage import log_class_usage

from .embedding_base import EmbeddingBase


class IntSingleCategoryEmbedding(EmbeddingBase):
    """Embed Dict of feature_id -> feature value to list of tensors (1 tensor per feature ID) after looking up feature value embedding,
    then apply optional pooling and MLP to final tensor.
    Passed in feature dict keys need to be in fixed order in forward.
    """

    class Config(ModuleConfig):
        embedding_dim: int = 32
        # mean / max / none (concat)
        pooling_type: str = "none"
        mlp_layer_dims: List[int] = []
        # Per feature buckets. emb bucket = mod(feature_value, feature_buckets[feature_id]).
        # When pooling_type is none, the concat order is based on the key order.
        feature_buckets: Dict[int, int] = {}

    @classmethod
    def from_config(cls, config: Config):
        """Factory method to construct an instance of DictEmbedding from
        the module's config object and the field's metadata object.

        Args:
            config (DictFeatConfig): Configuration object specifying all the
            parameters of DictEmbedding.
            metadata (FieldMeta): Object containing this field's metadata.

        Returns:
            type: An instance of DictEmbedding.

        """
        return cls(
            embedding_dim=config.embedding_dim,
            pooling_type=config.pooling_type,
            mlp_layer_dims=config.mlp_layer_dims,
            feature_buckets=config.feature_buckets,
        )

    def __init__(
        self,
        embedding_dim: int,
        pooling_type: str,
        mlp_layer_dims: List[int],
        feature_buckets: Dict[int, int],
    ) -> None:
        super().__init__(embedding_dim)
        self.pooling_type = pooling_type
        self.mlp_layer_dims = mlp_layer_dims
        self.num_intput_features = len(feature_buckets)
        input_dim = (
            self.num_intput_features * embedding_dim
            if self.pooling_type == "none"
            else embedding_dim
        )
        self.mlp = nn.Sequential(
            *(
                nn.Sequential(nn.Linear(m, n), nn.ReLU())
                for m, n in zip(
                    [input_dim] + list(mlp_layer_dims),
                    mlp_layer_dims,
                )
            )
        )

        self.feature_buckets = {int(k): v for k, v in feature_buckets.items()}
        self.feature_embeddings = nn.ModuleDict(
            {str(k): nn.Embedding(v, embedding_dim) for k, v in feature_buckets.items()}
        )
        log_class_usage(__class__)

    def get_output_dim(self):
        if self.mlp_layer_dims:
            return self.mlp_layer_dims[-1]

        if self.pooling_type == "none":
            return self.num_intput_features * self.embedding_dim
        elif self.pooling_type == "mean":
            return self.embedding_dim
        elif self.pooling_type == "max":
            return self.embedding_dim
        else:
            raise RuntimeError(f"Pooling type {self.pooling_type} is unsupported.")

    def forward(self, feats: Dict[int, torch.Tensor]) -> torch.Tensor:
        embeddings: List[torch.Tensor] = []
        for k, buckets in self.feature_buckets.items():
            feat = feats[k]
            feats_remap = torch.remainder(feat, buckets)
            feat_emb: nn.Embedding = self.feature_embeddings[str(k)]
            embeddings.append(feat_emb(feats_remap))

        if self.pooling_type == "none":
            reduced_embeds = torch.cat(embeddings, dim=1)
        elif self.pooling_type == "mean":
            reduced_embeds = torch.sum(torch.stack(embeddings, dim=1), dim=1)
        elif self.pooling_type == "max":
            reduced_embeds, _ = torch.max(torch.stack(embeddings, dim=1), dim=1)
        else:
            raise RuntimeError(f"Pooling type {self.pooling_type} is unsupported.")

        return self.mlp(reduced_embeds)

#!/usr/bin/env python3

from typing import List, Tuple

import torch
import torch.nn as nn
from pytext.config.component import Component, ComponentType, create_module
from pytext.data import CommonMetadata
from pytext.utils import cuda_utils


class Model(nn.Module, Component):
    """
    Generic model class that depends on input
    embedding, representation and projection to produce predicitons.
    """

    __COMPONENT_TYPE__ = ComponentType.MODEL

    @classmethod
    def from_config(cls, model_config, feat_config, metadata: CommonMetadata):
        embedding = create_module(feat_config, metadata=metadata)
        representation = create_module(
            model_config.repr_config, embed_dim=embedding.embedding_dim
        )
        projection = create_module(
            model_config.proj_config,
            from_dim=representation.representation_dim,
            to_dim=next(iter(metadata.labels.values())).vocab_size,
        )
        return cls(embedding, representation, projection)

    def __init__(self, embedding, representation, projection) -> None:
        nn.Module.__init__(self)
        self.embedding = embedding
        self.representation = representation
        self.projection = projection

    def forward(
        self,
        tokens: torch.Tensor,
        tokens_lens: torch.Tensor,
        dict_feat: Tuple[torch.Tensor, ...] = None,
        cap_feat: Tuple[torch.Tensor, ...] = None,
        chars: torch.Tensor = None,
    ) -> List[torch.Tensor]:
        # tokens dim: (bsz, max_seq_len) -> token_emb dim: (bsz, max_seq_len, dim)
        token_emb = self.embedding(tokens, tokens_lens, dict_feat, cap_feat, chars)
        return cuda_utils.parallelize(
            DataParallelModel(self.projection, self.representation),
            (token_emb, tokens_lens),
        )  # (bsz, nclasses)


class DataParallelModel(nn.Module):
    def __init__(self, projection, representation):
        super().__init__()
        self.projection = projection
        self.representation = representation

    def forward(self, token_emb: torch.Tensor, tokens_lens: torch.Tensor):
        rep = self.representation(token_emb, tokens_lens)
        if not isinstance(rep, (list, tuple)):
            rep = [rep]

        return self.projection(*rep)

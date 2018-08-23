#!/usr/bin/env python3

import torch
import torch.nn as nn
from typing import Tuple, List
from pytext.config.component import Component, ComponentType
from pytext.models.configs import gen_embedding_config
from pytext.utils import cuda_utils


class Model(nn.Module, Component):
    """
    Generic model class that depends on input
    embedding, representation and projection to produce predicitons.
    """
    __COMPONENT_TYPE__ = ComponentType.MODEL

    @classmethod
    def from_config(cls, model_config, feat_config, **metadata):
        return cls(
            model_config,
            gen_embedding_config(feat_config, **metadata),
            **metadata,
        )

    def __init__(self, model_config) -> None:
        nn.Module.__init__(self)
        Component.__init__(self, model_config)

    def forward(
        self,
        tokens: torch.Tensor,
        tokens_lens: torch.Tensor,
        dict_feat: Tuple[torch.Tensor, ...] = None,
        cap_feat: Tuple[torch.Tensor, ...] = None,
        chars: torch.Tensor = None,
    ) -> List[torch.Tensor]:
        # tokens dim: (bsz, max_seq_len) -> token_emb dim: (bsz, max_seq_len, dim)
        token_emb = self.embedding(tokens, dict_feat, cap_feat, chars)
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

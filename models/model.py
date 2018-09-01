#!/usr/bin/env python3

from typing import Dict, List, Tuple

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
            model_config.representation, embed_dim=embedding.embedding_dim
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

    def get_model_params_for_optimizer(
        self
    ) -> Tuple[Dict[str, nn.Parameter], Dict[str, nn.Parameter]]:
        """
        Current implementation makes a distinction between embedding params and
        rest of the model params because sparse gradients are supported for
        embeddings only.
        """
        prefix = "embedding"  # It is the name of embedding member variable.
        embedding_modules = self.embedding.__dict__.get("_modules")
        sparse_grad_embedding_param_names = set()
        for module_name, embedding_module in embedding_modules.items():
            if hasattr(embedding_module, "sparse") and embedding_module.sparse:
                for name, _ in embedding_module.named_parameters():
                    param_name = "{}.{}.{}".format(prefix, module_name, name)
                    sparse_grad_embedding_param_names.add(param_name)

        sparse_grads_params: Dict[str, nn.Parameter] = {}
        dense_grads_params: Dict[str, nn.Parameter] = {}
        for name, param in self.named_parameters():
            if name in sparse_grad_embedding_param_names:
                sparse_grads_params[name] = param
            else:
                dense_grads_params[name] = param

        return sparse_grads_params, dense_grads_params


class DataParallelModel(nn.Module):
    def __init__(self, projection, representation):
        super().__init__()
        self.projection = projection
        self.representation = representation

    def forward(self, token_emb: torch.Tensor, tokens_lens: torch.Tensor):
        rep = self.representation(token_emb, tokens_lens)
        if not isinstance(rep, (list, tuple)):
            rep = [rep]
        elif isinstance(rep[-1], tuple):
            # since some lstm based representations return states as (h0, c0)
            rep = rep[:-1]

        return self.projection(*rep)

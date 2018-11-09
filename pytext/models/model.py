#!/usr/bin/env python3

import os
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from pytext.config.component import Component, ComponentType, create_module
from pytext.data import CommonMetadata
from pytext.utils import cuda_utils

from .embeddings import EmbeddingList, EmbeddingBase


class Model(nn.Module, Component):
    """
    Generic model class that depends on input
    embedding, representation and decoder to produce predicitons.
    """

    __COMPONENT_TYPE__ = ComponentType.MODEL

    @classmethod
    def create_sub_embs(cls, emb_config, metadata: CommonMetadata):
        sub_embs = {}
        for name, config in emb_config._asdict().items():
            if issubclass(getattr(config, "__COMPONENT__", object), EmbeddingBase):
                sub_embs[name] = create_module(config, meta=metadata.features[name])
            else:
                print(f"{name} is not a config of embedding, skipping")
        return sub_embs

    @classmethod
    def compose_embedding(cls, sub_embs):
        return EmbeddingList(sub_embs.values(), concat=True)

    @classmethod
    def create_embedding(cls, emb_config, metadata: CommonMetadata):
        sub_embs = cls.create_sub_embs(emb_config, metadata)
        emb = cls.compose_embedding(sub_embs)
        if getattr(emb_config, "load_path", None):
            print(f"Loading state of embedding from {emb_config.load_path} ...")
            emb.load_state_dict(torch.load(emb_config.load_path))
        return emb

    @classmethod
    def from_config(cls, config, feat_config, metadata: CommonMetadata):
        embedding = cls.create_embedding(feat_config, metadata)
        # TODO hacky way to enable saving embedding now
        embedding.config = feat_config
        representation = create_module(
            config.representation, embed_dim=embedding.embedding_dim
        )
        decoder = create_module(
            config.decoder,
            from_dim=representation.representation_dim,
            to_dim=next(iter(metadata.labels.values())).vocab_size,
        )
        output_layer = create_module(config.output_layer, metadata)
        return cls(embedding, representation, decoder, output_layer)

    def save_modules(self, base_path: str = "", suffix: str = ""):
        for module in [self.embedding, self.representation, self.decoder]:
            if getattr(module.config, "save_path", None):
                path = module.config.save_path + suffix
                if base_path:
                    path = os.path.join(base_path, path)
                print(f"Saving state of module {type(module).__name__} to {path} ...")
                torch.save(module.state_dict(), path)

    def __init__(self, embedding, representation, decoder, output_layer) -> None:
        nn.Module.__init__(self)
        self.embedding = embedding
        self.representation = representation
        self.decoder = decoder
        self.output_layer = output_layer

    def get_loss(self, logit, target, context):
        return self.output_layer.get_loss(logit, target, context)

    def get_pred(self, logit, target, context, *args):
        return self.output_layer.get_pred(logit, target, context)

    def forward(self, *inputs) -> List[torch.Tensor]:
        embedding_input = inputs[: self.embedding.num_emb]
        token_emb = self.embedding(*embedding_input)
        other_input = inputs[self.embedding.num_emb :]
        return cuda_utils.parallelize(
            DataParallelModel(self.representation, self.decoder),
            (token_emb, *other_input),
        )  # returned Tensor's dim = (batch_size, num_classes)

    def get_model_params_for_optimizer(
        self
    ) -> Tuple[Dict[str, nn.Parameter], Dict[str, nn.Parameter]]:
        """
        Current implementation makes a distinction between embedding params and
        rest of the model params because sparse gradients are supported for
        embeddings only.
        """
        prefix = "embedding"  # It is the name of embedding member variable.
        sparse_grad_embedding_param_names = set()
        for module_name, embedding_module in self.embedding.named_children():
            if getattr(embedding_module, "sparse", False):
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
    def __init__(self, representation, decoder):
        super().__init__()
        self.representation = representation
        self.decoder = decoder

    def forward(self, *inputs):
        input_representation = self.representation(*inputs)
        if not isinstance(input_representation, (list, tuple)):
            input_representation = [input_representation]
        elif isinstance(input_representation[-1], tuple):
            # since some lstm based representations return states as (h0, c0)
            input_representation = input_representation[:-1]
        return self.decoder(*input_representation)

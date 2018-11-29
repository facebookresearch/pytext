#!/usr/bin/env python3

import os
from typing import List

import torch
import torch.nn as nn
from pytext.config.component import Component, ComponentType
from pytext.data import CommonMetadata
from pytext.models.module import create_module

from .embeddings import EmbeddingBase, EmbeddingList


class Model(nn.Module, Component):
    """
    Generic model class that depends on input
    embedding, representation and decoder to produce predicitons.
    """

    __EXPANSIBLE__ = True
    __COMPONENT_TYPE__ = ComponentType.MODEL

    @classmethod
    def create_sub_embs(cls, emb_config, metadata: CommonMetadata):
        sub_embs = {}
        for name, config in emb_config._asdict().items():
            if issubclass(getattr(config, "__COMPONENT__", object), EmbeddingBase):
                sub_embs[name] = create_module(config, metadata=metadata.features[name])
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
        emb.config = emb_config
        return emb

    @classmethod
    def from_config(cls, config, feat_config, metadata: CommonMetadata):
        embedding = create_module(
            feat_config, create_fn=cls.create_embedding, metadata=metadata
        )
        representation = create_module(
            config.representation, embed_dim=embedding.embedding_dim
        )
        decoder = create_module(
            config.decoder,
            in_dim=representation.representation_dim,
            out_dim=metadata.target.vocab_size,
        )
        output_layer = create_module(config.output_layer, metadata.target)
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

    def contextualize(self, context):
        self.context = context

    def get_loss(self, logit, target, context):
        return self.output_layer.get_loss(logit, target, context)

    def get_pred(self, logit, target, context, *args):
        return self.output_layer.get_pred(logit, target, context)

    def forward(self, *inputs) -> List[torch.Tensor]:
        embedding_input = inputs[: self.embedding.num_emb_modules]
        token_emb = self.embedding(*embedding_input)
        other_input = inputs[self.embedding.num_emb_modules :]
        input_representation = self.representation(token_emb, *other_input)
        if not isinstance(input_representation, (list, tuple)):
            input_representation = [input_representation]
        elif isinstance(input_representation[-1], tuple):
            # since some lstm based representations return states as (h0, c0)
            input_representation = input_representation[:-1]
        return self.decoder(
            *input_representation
        )  # returned Tensor's dim = (batch_size, num_classes)

    def prepare_for_onnx_export_(self, **kwargs):
        """Make model exportable via ONNX trace."""

        def apply_prepare_for_onnx_export_(module):
            if module != self and hasattr(module, "prepare_for_onnx_export_"):
                module.prepare_for_onnx_export_(**kwargs)

        self.apply(apply_prepare_for_onnx_export_)

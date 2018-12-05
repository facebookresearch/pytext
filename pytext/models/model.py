#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
from typing import Dict, List

import torch
import torch.nn as nn
from pytext.common.constants import Stage
from pytext.config.component import Component, ComponentType
from pytext.data import CommonMetadata
from pytext.models.module import create_module

from .embeddings import EmbeddingBase, EmbeddingList


class Model(nn.Module, Component):
    """
    Generic model class that depends on input
    embedding, representation and decoder to produce predicitons.

    Model also have a stage flag to indictate it's in Train, eval, or test stage.
    This is because the builtin train/evel flag in PyTorch can't distinguish evel
    and test, which is required to support some use cases
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

    def __init__(
        self, embedding, representation, decoder, output_layer, stage=Stage.TRAIN
    ) -> None:
        nn.Module.__init__(self)

        self.embedding = embedding
        self.representation = representation
        self.decoder = decoder
        self.output_layer = output_layer
        self.stage = stage

    def train(self, mode=True):
        """
        Override to set stage
        """
        super().train(mode)
        self.stage = Stage.TRAIN

    def eval(self, stage=Stage.TEST):
        """
        Override to set stage
        """
        super().eval()
        self.stage = stage

    def contextualize(self, context):
        self.context = context

    def get_loss(self, logit, target, context):
        return self.output_layer.get_loss(logit, target, context)

    def get_pred(self, logit, target=None, context=None, *args):
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

    def get_param_groups_for_optimizer(self) -> List[Dict[str, List[nn.Parameter]]]:
        """
        Returns a list of parameter groups of the format {"params": param_list}.
        The parameter groups loosely correspond to layers and are ordered from low
        to high. Currently, only the embedding layer can provide multiple param groups,
        and other layers are put into one param group. The output of this method
        is passed to the optimizer so that schedulers can change learning rates
        by layer.
        """
        non_emb_params = dict(self.named_parameters())
        model_params = [non_emb_params]

        # some subclasses of Model (e.g. Ensemble) do not have embeddings
        embedding = getattr(self, "embedding", None)
        if embedding is not None:
            emb_params_by_layer = self.embedding.get_param_groups_for_optimizer()

            # Delete params from the embedding layers
            for emb_params in emb_params_by_layer:
                for name in emb_params:
                    del non_emb_params["embedding.%s" % name]

            model_params = emb_params_by_layer + model_params
            print_str = (
                "Model has %d param groups (%d from embedding module) for optimizer"
            )
            print(print_str % (len(model_params), len(emb_params_by_layer)))

        model_params = [{"params": params.values()} for params in model_params]
        return model_params

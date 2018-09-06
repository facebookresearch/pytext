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
        output_layer = create_module(model_config.output_config, metadata)
        return cls(embedding, representation, projection, output_layer)

    def __init__(self, embedding, representation, projection, output_layer) -> None:
        nn.Module.__init__(self)
        self.embedding = embedding
        self.representation = representation
        self.projection = projection
        self.output_layer = output_layer

    def get_loss(self, logit, target, context):
        return self.output_layer.get_loss(logit, target, context)

    def get_pred(self, logit, context):
        return self.output_layer.get_pred(logit, context)

    def forward(self, *inputs) -> List[torch.Tensor]:
        token_emb = self.embedding(*inputs)
        return cuda_utils.parallelize(
            DataParallelModel(self.projection, self.representation),
            (token_emb, *inputs[1:]),  # Assumption: inputs[0] = tokens
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

    def forward(self, *inputs):
        input_representation = self.representation(*inputs)
        if not isinstance(input_representation, (list, tuple)):
            input_representation = [input_representation]
        elif isinstance(input_representation[-1], tuple):
            # since some lstm based representations return states as (h0, c0)
            input_representation = input_representation[:-1]

        return self.projection(*input_representation)

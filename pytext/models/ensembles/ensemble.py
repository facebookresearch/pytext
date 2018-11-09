#!/usr/bin/env python3
from copy import deepcopy
from typing import Any, Dict, List, Tuple

import torch.nn as nn
from pytext.config import ConfigBase
from pytext.config.component import create_model
from pytext.models.model import Model


class Ensemble(Model):
    class Config(ConfigBase):
        models: List[Any]
        sample_rate: float = 1.0

    def __init__(self, config, models, *arg, **kwargs):
        nn.Module.__init__(self)
        self.models = nn.ModuleList(models)
        self.output_layer = deepcopy(models[0].output_layer)

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def merge_sub_models(self):
        pass

    def cpu(self):
        # Needed when the ensembled models are wrapped in DistributedModel
        for i in range(len(self.models)):
            self.models[i] = self.models[i].cpu()
        return self

    @classmethod
    def from_config(cls, model_config, feat_config, *arg, **kwargs):
        sub_models = [
            create_model(sub_model_config, feat_config, *arg, **kwargs)
            for sub_model_config in model_config.models
        ]
        return cls(model_config, sub_models, *arg, **kwargs)

    def get_model_params_for_optimizer(
        self
    ) -> Tuple[Dict[str, nn.Parameter], Dict[str, nn.Parameter]]:
        sparse_grads_params: Dict[str, nn.Parameter] = {}
        dense_grads_params: Dict[str, nn.Parameter] = {}
        for i, model in enumerate(self.models):
            s, d = self._get_single_model_params_for_optimizer(model, i)
            sparse_grads_params.update(s)
            dense_grads_params.update(d)
        return sparse_grads_params, dense_grads_params

    def save_modules(self, base_path: str = "", suffix: str = ""):
        for model in self.models:
            model.save_modules(base_path, suffix)

    def _get_single_model_params_for_optimizer(
        self, model: nn.Module, model_id: int
    ) -> Tuple[Dict[str, nn.Parameter], Dict[str, nn.Parameter]]:
        """
        Current implementation makes a distinction between embedding params and
        rest of the model params because sparse gradients are supported for
        embeddings only.
        """
        sparse_grad_embedding_param_names = set()
        if hasattr(model, "embedding"):
            prefix = "embedding"  # It is the name of embedding member variable.
            embedding_modules = model.embedding.__dict__.get("_modules")
            for module_name, embedding_module in embedding_modules.items():
                if hasattr(embedding_module, "sparse") and embedding_module.sparse:
                    for name, _ in embedding_module.named_parameters():
                        param_name = "{}.{}.{}".format(prefix, module_name, name)
                        sparse_grad_embedding_param_names.add(param_name)

        sparse_grads_params: Dict[str, nn.Parameter] = {}
        dense_grads_params: Dict[str, nn.Parameter] = {}
        for name, param in model.named_parameters():
            key = "{}_{}".format(model_id, name)
            if name in sparse_grad_embedding_param_names:
                sparse_grads_params[key] = param
            else:
                dense_grads_params[key] = param

        return sparse_grads_params, dense_grads_params

#!/usr/bin/env python3
from typing import Any, List

import torch.nn as nn
from pytext.common.registry import create_model
from pytext.config import ConfigBase


class EnsembleModelConfig(ConfigBase):
    models: List[Any]
    sample_rate: float = 1.0


class Ensemble(nn.Module):
    def __init__(self, config, models, **metadata):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    @classmethod
    def from_config(cls, model_config, feat_config, **metadata):
        sub_models = [
            create_model(sub_model_config, feat_config, **metadata)
            for sub_model_config in model_config.models
        ]
        return cls(model_config, sub_models, **metadata)

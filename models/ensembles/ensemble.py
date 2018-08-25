#!/usr/bin/env python3
from typing import Any, List

import torch.nn as nn
from pytext.config.component import create_model
from pytext.config import ConfigBase
from pytext.models.model import Model


class Ensemble(Model):
    class Config(ConfigBase):
        models: List[Any]
        sample_rate: float = 1.0

    def __init__(self, config, models, *arg, **kwargs):
        nn.Module.__init__(self)
        self.models = nn.ModuleList(models)

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    @classmethod
    def from_config(cls, model_config, feat_config, *arg, **kwargs):
        sub_models = [
            create_model(sub_model_config, feat_config, *arg, **kwargs)
            for sub_model_config in model_config.models
        ]
        return cls(model_config, sub_models, *arg, **kwargs)

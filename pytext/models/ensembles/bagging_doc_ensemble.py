#!/usr/bin/env python3
from typing import List

import torch
from pytext.config import ConfigBase
from pytext.models.doc_model import DocModel

from .ensemble import Ensemble


class BaggingDocEnsemble(Ensemble):
    class Config(Ensemble.Config, ConfigBase):
        models: List[DocModel.Config]

    def forward(self, *args, **kwargs):
        logit_d_list = torch.cat(
            tuple(model.forward(*args, **kwargs).unsqueeze(2) for model in self.models),
            dim=2,
        )

        return torch.mean(logit_d_list, dim=2)

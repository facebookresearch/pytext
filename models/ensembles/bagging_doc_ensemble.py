#!/usr/bin/env python3
from typing import List, Union

import torch
from pytext.config import ConfigBase
from pytext.models.doc_models import DocBLSTM, DocNN
from .ensemble import Ensemble


class BaggingDocEnsemble(Ensemble):
    class Config(Ensemble.Config, ConfigBase):
        models: List[Union[DocBLSTM.Config, DocNN.Config]]

    def forward(self, *args, **kwargs):
        logit_d_list = None
        for model in self.models:
            [logit_d] = model.forward(*args, **kwargs)
            logit_d = logit_d.unsqueeze(2)

            if logit_d_list is None:
                logit_d_list = logit_d
            else:
                logit_d_list = torch.cat([logit_d_list, logit_d], dim=2)

        return [torch.mean(logit_d_list, dim=2)]

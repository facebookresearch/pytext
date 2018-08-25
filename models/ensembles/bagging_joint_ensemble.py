#!/usr/bin/env python3
from typing import List

import torch
from pytext.common.constants import DatasetFieldName
from pytext.config import ConfigBase
from pytext.models.crf import CRF
from pytext.models.joint_model import JointModel

from .ensemble import Ensemble


class BaggingJointEnsemble(Ensemble):
    class Config(Ensemble.Config, ConfigBase):
        models: List[JointModel.Config]
        use_crf: bool = False

    def __init__(self, config, models, metadata):
        super().__init__(config, models)
        word_label_num = metadata.labels[DatasetFieldName.WORD_LABEL_FIELD].vocab_size
        if config.use_crf:
            self.crf_transition_matrices = []
            self.crf = CRF(word_label_num)

    def forward(self, *args, **kwargs):
        logit_d_list, logit_w_list = None, None
        for model in self.models:
            logit_d, logit_w = model.forward(*args, **kwargs)
            logit_d, logit_w = logit_d.unsqueeze(2), logit_w.unsqueeze(3)

            if logit_d_list is None:
                logit_d_list = logit_d
            else:
                logit_d_list = torch.cat([logit_d_list, logit_d], dim=2)

            if logit_w_list is None:
                logit_w_list = logit_w
            else:
                logit_w_list = torch.cat([logit_w_list, logit_w], dim=3)

        return [torch.mean(logit_d_list, dim=2), torch.mean(logit_w_list, dim=3)]

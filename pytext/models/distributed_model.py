#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch.nn as nn
from pytext.common.constants import Stage


class DistributedModel(nn.parallel.DistributedDataParallel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getattr__(self, name):
        wrapped_module = super().__getattr__("module")
        if hasattr(wrapped_module, name):
            return getattr(wrapped_module, name)
        return super().__getattr__(name)

    def cpu(self):
        wrapped_module = super().__getattr__("module")
        return wrapped_module.cpu()

    def train(self, mode=True):
        """
        Override to set stage
        """
        # use DistributedDataParallel.train since it fits distributed_training
        super().train(mode)
        self._set_module_stage(Stage.TRAIN)

    def eval(self, stage=Stage.TEST):
        """
        Override to set stage
        """
        # use DistributedDataParallel.eval since it fits distributed_training
        super().eval()
        self._set_module_stage(stage)

    def _set_module_stage(self, stage):
        wrapped_module = super().__getattr__("module")
        if hasattr(wrapped_module, "stage"):
            wrapped_module.stage = stage

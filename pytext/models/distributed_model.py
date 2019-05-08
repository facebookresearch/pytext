#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch.nn as nn
from pytext.common.constants import Stage


class DistributedModel(nn.parallel.DistributedDataParallel):
    """
    Wrapper model class to train models in distributed data parallel manner.
    The way to use this class to train your module in distributed manner is::

        distributed_model = DistributedModel(
            module=model,
            device_ids=[device_id0, device_id1],
            output_device=device_id0,
            broadcast_buffers=False,
        )


    where, `model` is the object of the actual model class you want to train in
    distributed manner.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.accumulate_grads = False

    def __getattr__(self, name):
        wrapped_module = super().__getattr__("module")
        if hasattr(wrapped_module, name):
            return getattr(wrapped_module, name)
        return super().__getattr__(name)

    def cpu(self):
        wrapped_module = super().__getattr__("module")
        return wrapped_module.cpu()

    def state_dict(self, *args, **kwargs):
        wrapped_module = super().__getattr__("module")
        return wrapped_module.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        wrapped_module = super().__getattr__("module")
        return wrapped_module.load_state_dict(*args, **kwargs)

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

    def accumulate_gradients(self, enable):
        self.accumulate_grads = enable

    def forward(self, *inputs, **kwargs):
        # support accumulate gradients in PyText
        if self.accumulate_grads:
            return self.module(*inputs, **kwargs)
        else:
            return super().forward(*inputs, **kwargs)

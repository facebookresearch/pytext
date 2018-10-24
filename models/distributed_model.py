#!/usr/bin/env python3

import torch.nn as nn


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

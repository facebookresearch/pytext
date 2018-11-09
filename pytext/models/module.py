#!/usr/bin/env python3

import torch.nn as nn
import torch
from pytext.config.component import Component, ComponentType, create_component
from pytext.config.module_config import ModuleConfig


def _create_module_from_registry(module_config, *args, **kwargs):
    return create_component(ComponentType.MODULE, module_config, *args, **kwargs)


def create_module(
    module_config, *args, create_fn=_create_module_from_registry, **kwargs
):
    module = create_fn(module_config, *args, **kwargs)
    name = type(module).__name__
    if getattr(module_config, "load_path", None):
        print(f"Loading state of module {name} from {module_config.load_path} ...")
        module.load_state_dict(torch.load(module_config.load_path))
    if getattr(module_config, "freeze", False):
        print(f"Freezing the parameters of module {name} ...")
        for param in module.parameters():
            param.requires_grad = False
    return module


class Module(nn.Module, Component):
    """
    Generic model class that depends on input
    embedding, representation and decoder to produce predictions.
    """

    Config = ModuleConfig

    __COMPONENT_TYPE__ = ComponentType.MODULE

    def __init__(self, config=None) -> None:
        nn.Module.__init__(self)
        Component.__init__(self, config)

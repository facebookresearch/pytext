#!/usr/bin/env python3

from typing import Dict

import torch
import torch.nn as nn
from pytext.config.component import Component, ComponentType, create_component
from pytext.config.module_config import ModuleConfig


SHARED_MODULE_REGISTRY: Dict[str, torch.nn.Module] = {}


def _create_module_from_registry(module_config, *args, **kwargs):
    return create_component(ComponentType.MODULE, module_config, *args, **kwargs)


def create_module(
    module_config, *args, create_fn=_create_module_from_registry, **kwargs
):
    # the first module with a given shared_module_key and type is saved in
    # SHARED_MODULE_REGISTRY.  The rest will reuse the saved module and thus
    # share parameters.
    shared_module_key = getattr(module_config, "shared_module_key", None)
    module = SHARED_MODULE_REGISTRY.get(
        (shared_module_key, type(module_config)),
        create_fn(module_config, *args, **kwargs),
    )
    name = type(module).__name__
    if getattr(module_config, "load_path", None):
        print(f"Loading state of module {name} from {module_config.load_path} ...")
        module.load_state_dict(torch.load(module_config.load_path))
    if getattr(module_config, "freeze", False):
        print(f"Freezing the parameters of module {name} ...")
        for param in module.parameters():
            param.requires_grad = False
    if shared_module_key:
        SHARED_MODULE_REGISTRY[(shared_module_key, type(module_config))] = module
    return module


class Module(nn.Module, Component):
    """Generic module class that serves as base class for all PyText modules."""

    Config = ModuleConfig

    __COMPONENT_TYPE__ = ComponentType.MODULE

    def __init__(self, config=None) -> None:
        nn.Module.__init__(self)
        Component.__init__(self, config)

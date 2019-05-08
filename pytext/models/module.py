#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

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
    """Create module object given the module's config object. It depends on the
    global shared module registry. Hence, your module must be available for the
    registry. This entails that your module must be imported somewhere in the
    code path during module creation (ideally in your model class) for the module
    to be visible for registry.

    Args:
        module_config (type): Module config object.
        create_fn (type): The function to use for creating the module. Use this
            parameter if your module creation requires custom code and pass your
            function here. Defaults to `_create_module_from_registry()`.

    Returns:
        type: Description of returned object.

    """
    # the first module with a given shared_module_key and type is saved in
    # SHARED_MODULE_REGISTRY.  The rest will reuse the saved module and thus
    # share parameters.
    shared_module_key = getattr(module_config, "shared_module_key", None)
    try:
        module = SHARED_MODULE_REGISTRY[(shared_module_key, type(module_config))]
    except KeyError:
        module = create_fn(module_config, *args, **kwargs)

    name = type(module).__name__
    if getattr(module_config, "load_path", None):
        print(f"Loading state of module {name} from {module_config.load_path} ...")
        module.load_state_dict(torch.load(module_config.load_path, map_location="cpu"))
    if getattr(module_config, "freeze", False):
        print(f"Freezing the parameters of module {name} ...")
        module.freeze()
    if shared_module_key:
        SHARED_MODULE_REGISTRY[(shared_module_key, type(module_config))] = module
    return module


class Module(nn.Module, Component):
    """Generic module class that serves as base class for all PyText modules.

    Args:
        config (type): Module's `config` object. Specific contents of this object
            depends on the module. Defaults to None.

    """

    Config = ModuleConfig

    __COMPONENT_TYPE__ = ComponentType.MODULE

    def __init__(self, config=None) -> None:
        nn.Module.__init__(self)
        Component.__init__(self, config)

    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import zipfile
from typing import Dict

import torch
import torch.jit
import torch.nn as nn
from pytext.config.component import Component, ComponentType, create_component
from pytext.config.module_config import ModuleConfig
from pytext.config.pytext_config import ConfigBase


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
    typed_shared_module_key = (shared_module_key, type(module_config))
    load_path = getattr(module_config, "load_path", None)
    is_torchscript_load_path = load_path and zipfile.is_zipfile(load_path)
    module = SHARED_MODULE_REGISTRY.get(typed_shared_module_key)
    if not module:
        if is_torchscript_load_path:
            with open(load_path, "rb") as load_file:
                module = torch.jit.load(load_file)
        else:
            module = create_fn(module_config, *args, **kwargs)

    name = type(module).__name__
    if load_path and not is_torchscript_load_path:
        print(f"Loading state of module {name} from {load_path} ...")
        module.load_state_dict(torch.load(load_path, map_location="cpu"))
    if getattr(module_config, "freeze", False):
        print(f"Freezing the parameters of module {name} ...")
        module.freeze()
    if shared_module_key:
        SHARED_MODULE_REGISTRY[typed_shared_module_key] = module
    module.save_path = getattr(module_config, "save_path", None)
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

    def __dir__(self):
        """Jit doesnt allow scripting of attributes whose classname includes "."

        Example Repro:

        class OldModule(Module):
            class Config(ConfigBase):
                a: int = 5

            @classmethod
            def from_config(cls, config: Config):
                return cls(config.a)

            def __init__(self, a):
                super().__init__()
                self.a = a

            def forward(self, b: int) -> int:
                return b + self.a

        m = OldModule.from_config(OldModule.Config())
        jit.script(m)

            > RuntimeError: Could not get qualified name for class 'OldModule.Config':
        'OldModule.Config' is not a valid identifier

        print(m.Config.__name__)
            > OldModule.Config

        At the sametime, we dont need to script the config classes because they
        are not needed during inference time. Hence in this workaround we skip
        the config classes.

        Ideal solution is that when building models they should be inheriting
        from nn.Module only and not Component. This requires significant changes
        to the way models are created in PyText.

        """
        result = super().__dir__()
        return [
            r
            for r in result
            if not (
                isinstance(getattr(self, r, None), type)
                and issubclass(getattr(self, r, None), ConfigBase)
            )
        ]

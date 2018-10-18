#!/usr/bin/env python3

import torch.nn as nn
from pytext.config.component import Component, ComponentType
from pytext.config.module_config import ModuleConfig


class Module(nn.Module, Component):
    """
    Generic model class that depends on input
    embedding, representation and decoder to produce predicitons.
    """

    Config = ModuleConfig

    __COMPONENT_TYPE__ = ComponentType.MODULE

    def __init__(self, config=None) -> None:
        nn.Module.__init__(self)
        Component.__init__(self, config)

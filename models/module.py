#!/usr/bin/env python3

from typing import Optional

import torch.nn as nn
from pytext.config import ConfigBase
from pytext.config.component import Component, ComponentType


class Module(nn.Module, Component):
    """
    Generic model class that depends on input
    embedding, representation and decoder to produce predicitons.
    """

    class Config(ConfigBase):
        # checkpoint loading/saving paths
        load_path: Optional[str] = None
        save_path: Optional[str] = None

    __COMPONENT_TYPE__ = ComponentType.MODULE

    def __init__(self, config=None) -> None:
        nn.Module.__init__(self)
        Component.__init__(self, config)

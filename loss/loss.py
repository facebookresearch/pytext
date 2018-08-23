#!/usr/bin/env python3

from pytext.config.component import Component, ComponentType


class Loss(Component):
    """Base class for loss functions"""
    __COMPONENT_TYPE__ = ComponentType.LOSS

    def __init__(self, config=None, **metadata):
        super().__init__(config)

    def loss(self, m_out, targets, model=None, context=None, reduce: bool=True):
        raise NotImplementedError

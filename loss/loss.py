#!/usr/bin/env python3

from abc import ABC, abstractmethod


class Loss(ABC):
    """Base class for loss functions"""
    def __init__(self, config, **kwargs):
        pass

    @abstractmethod
    def loss(self, m_out, targets, model=None, context=None, reduce: bool=True):
        pass

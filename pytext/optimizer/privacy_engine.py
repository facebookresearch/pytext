#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Optional

import torchdp
from pytext.config import ConfigBase
from pytext.config.component import Component, ComponentType


class PrivacyEngine(Component):
    """
    A wrapper around PrivacyEngine of pytorch-dp
    """

    __COMPONENT_TYPE__ = ComponentType.PRIVACY_ENGINE
    __EXPANSIBLE__ = False

    class Config(ConfigBase):
        enabled: bool = False
        noise_multiplier: float = 1.0
        max_grad_norm: float = 1.0
        batch_size: float = 128
        dataset_size: float = 128000
        target_delta: Optional[float] = 0.000001

    def __init__(
        self,
        model,
        enabled,
        noise_multiplier,
        max_grad_norm,
        batch_size,
        dataset_size,
        target_delta=0.000001,
    ):
        self.enabled = enabled
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self.target_delta = target_delta

        if self.enabled:
            self._privacy_engine = torchdp.PrivacyEngine(
                model,
                self.batch_size,
                self.dataset_size,
                alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
                noise_multiplier=self.noise_multiplier,
                max_grad_norm=self.max_grad_norm,
                target_delta=self.target_delta,
            )

    @classmethod
    def from_config(cls, config: Config, model):
        return cls(
            model=model,
            enabled=config.enabled,
            noise_multiplier=config.noise_multiplier,
            max_grad_norm=config.max_grad_norm,
            batch_size=config.batch_size,
            dataset_size=config.dataset_size,
            target_delta=config.target_delta,
        )

    def attach(self, optimizer):
        if self.enabled:
            self._privacy_engine.attach(optimizer)

    def detach(self):
        if self.enabled:
            self._privacy_engine.detach()

    def get_privacy_spent(self):
        if self.enabled:
            return self._privacy_engine.get_privacy_spent()
        return (None, None)

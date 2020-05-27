#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List, Optional

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
        noise_multiplier: float
        max_grad_norm: float
        batch_size: float
        dataset_size: float
        target_delta: Optional[float] = 0.000001
        alphas: Optional[List[float]] = [1 + x / 10.0 for x in range(1, 100)] + list(
            range(12, 64)
        )

    def __init__(
        self,
        model,
        optimizer,
        noise_multiplier,
        max_grad_norm,
        batch_size,
        dataset_size,
        target_delta,
        alphas,
    ):
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self.target_delta = target_delta
        self.alphas = alphas

        self._privacy_engine = torchdp.PrivacyEngine(
            model,
            self.batch_size,
            self.dataset_size,
            self.alphas,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm,
            target_delta=self.target_delta,
        )
        self._privacy_engine.attach(optimizer)

    @classmethod
    def from_config(cls, config: Config, model, optimizer):
        return cls(
            model=model,
            optimizer=optimizer,
            noise_multiplier=config.noise_multiplier,
            max_grad_norm=config.max_grad_norm,
            batch_size=config.batch_size,
            dataset_size=config.dataset_size,
            target_delta=config.target_delta,
            alphas=config.alphas,
        )

    def attach(self, optimizer):
        self._privacy_engine.attach(optimizer)

    def detach(self):
        self._privacy_engine.detach()

    def get_privacy_spent(self):
        return self._privacy_engine.get_privacy_spent()

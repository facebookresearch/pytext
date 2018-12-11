#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Generic, TypeVar

from pytext.config.module_config import ModuleConfig
from pytext.models import Model


class SameModelBag(ModuleConfig):
    """Class representing a bag of duplicate models with the same config.
    Attributes:
        count (int): Size of the bag/number of models.
        model (Model.Config): The config for this model.
    """

    count: int = 1
    model: Model.Config

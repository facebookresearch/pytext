#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .model import BaseModel, Model
from .two_tower_classification_model import TwoTowerClassificationModel


__all__ = ["Model", "BaseModel", "TwoTowerClassificationModel"]

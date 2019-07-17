#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from enum import Enum
from typing import List, Optional

from .pytext_config import ConfigBase


class ModuleConfig(ConfigBase):
    # Checkpoint load path
    load_path: Optional[str] = None
    # Checkpoint save path, relative to PyTextConfig.modules_save_dir (if set)
    save_path: Optional[str] = None
    # Freezing a module means its parameters won't be updated during training.
    freeze: bool = False
    # modules which have the same shared_module_key and type share parameters
    shared_module_key: Optional[str] = None


class CNNParams(ConfigBase):
    # Number of feature maps for each kernel
    kernel_num: int = 100
    # Kernel sizes to use in convolution
    kernel_sizes: List[int] = [3, 4]


class PoolingType(Enum):
    MEAN = "mean"
    MAX = "max"


class SlotAttentionType(Enum):
    NO_ATTENTION = "no_attention"
    CONCAT = "concat"
    MULTIPLY = "multiply"
    DOT = "dot"

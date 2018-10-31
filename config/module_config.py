#!/usr/bin/env python3
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

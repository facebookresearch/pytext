#!/usr/bin/env python3
from enum import Enum
from typing import List

from .pytext_config import ConfigBase


class LSTMParams(ConfigBase):
    # The number of features in the lstm hidden state
    lstm_dim: int = 100
    num_layers: int = 1


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

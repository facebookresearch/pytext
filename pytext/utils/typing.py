#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from enum import Enum


class WeightingMethod(Enum):
    CLASS_RATIO = "CLASS_RATIO"  # weight = #neg/#pos for each class.
    SQRT_RATIO = "SQRT"  # normalized by square root of CLASS_RATIO
    CAPPED_RATIO = "CAP"  # weight = # avg positive / # positive if # positive is greater than average, otherwise 1.0

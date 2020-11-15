#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved#!/usr/bin/env python3

from typing import Dict


# Keep in sync with QuantizationType enum in textray_contants.thrift
# We can't directly use the thrift enum because TorchScript doesn't support it
QUANTIZATION_SCHEMA: Dict[str, int] = {"NONE": 1, "INT1_QUANTIZATION": 2}

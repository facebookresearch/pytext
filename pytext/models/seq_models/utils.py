#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Optional

import torch


def prepare_full_key(instance_id: str, key: str, secondary_key: Optional[str] = None):
    if secondary_key is not None:
        return instance_id + "." + key + "." + secondary_key
    else:
        return instance_id + "." + key


def make_positions(input, padding_idx: int):
    """Replace non-padding symbols with their position numbers.

    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    mask = input.ne(padding_idx)
    return torch.cumsum(mask, dim=1) * mask + padding_idx

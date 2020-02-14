#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Optional


def prepare_full_key(instance_id: str, key: str, secondary_key: Optional[str] = None):
    if secondary_key is not None:
        return instance_id + "." + key + "." + secondary_key
    else:
        return instance_id + "." + key

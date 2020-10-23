#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Any

from omegaconf import OmegaConf
from omegaconf.base import Container


def to_container(obj: Any):
    """
    Container types like list and dict converted from OmegaConfs are of
    types of OmegaConf's Containers. This breaks while intermixing them
    with code that will be converted to torchscript and breaks. This
    method will convert these types to native python types.
    """
    if isinstance(obj, Container):
        return OmegaConf.to_container(obj)
    return obj

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
from pytext.utils.usage import log_class_usage

from .representation_base import RepresentationBase


class PassThroughRepresentation(RepresentationBase):
    def __init__(self, config: RepresentationBase.Config, embed_dim: int) -> None:
        super().__init__(config)
        self.representation_dim = embed_dim
        log_class_usage(__class__)

    def forward(self, embedded_tokens: torch.Tensor, *args) -> torch.Tensor:
        return embedded_tokens

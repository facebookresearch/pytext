#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from pytext.models.module import Module


class RepresentationBase(Module):
    def __init__(self, config):
        super().__init__(config)
        self.representation_dim = None

    def forward(self, *inputs):
        raise NotImplementedError()

    def get_representation_dim(self):
        return self.representation_dim

    def _preprocess_inputs(self, inputs):
        raise NotImplementedError()

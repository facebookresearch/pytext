#!/usr/bin/env python3

import torch.nn as nn


class RepresentationBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.representation_dim = 0

    def forward(self, *input):
        raise NotImplementedError()

    def get_representation_dim(self):
        return self.representation_dim

    def _preprocess_inputs(self, inputs):
        raise NotImplementedError()

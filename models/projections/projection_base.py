#!/usr/bin/env python3

import torch.nn as nn


class ProjectionBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_dim = 0
        self.out_dim = 0

    def forward(self, *input):
        raise NotImplementedError()

    def get_projection(self):
        raise NotImplementedError()

    def get_in_dim(self):
        return self.in_dim

    def get_out_dim(self):
        return self.out_dim

#!/usr/bin/env python3

from pytext.models.module import Module


class DecoderBase(Module):
    def __init__(self, config):
        super().__init__(config)
        self.in_dim = 0
        self.out_dim = 0

    def forward(self, *input):
        raise NotImplementedError()

    def get_decoder(self):
        raise NotImplementedError()

    def get_in_dim(self):
        return self.in_dim

    def get_out_dim(self):
        return self.out_dim

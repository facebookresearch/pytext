#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from pytext.config import ConfigBase
from pytext.models.module import Module


class DecoderBase(Module):
    """Base class for all decoder modules.

    Args:
        config (ConfigBase): Configuration object.

    Attributes:
        in_dim (int): Dimension of input Tensor passed to the decoder.
        out_dim (int): Dimension of output Tensor produced by the decoder.

    """

    def __init__(self, config: ConfigBase):
        super().__init__(config)
        self.input_dim = 0
        self.target_dim = 0
        self.num_decoder_modules = 0

    def forward(self, *input):
        raise NotImplementedError()

    def get_decoder(self):
        """Returns the decoder module.
        """
        raise NotImplementedError()

    def get_in_dim(self) -> int:
        """Returns the dimension of the input Tensor that the decoder accepts.
        """
        return self.in_dim

    def get_out_dim(self) -> int:
        """Returns the dimension of the input Tensor that the decoder emits.
        """
        return self.out_dim

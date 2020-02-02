#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Dict, Optional

from pytext.config import ConfigBase
from pytext.models.module import Module
from torch import Tensor, nn

from .utils import prepare_full_key


global_counter = 0


class PyTextSeq2SeqModule(Module):
    instance_id: str = None

    def __init__(self):
        super().__init__()
        self.assign_id()

    def assign_id(self):
        global global_counter
        self.instance_id = ".".join([type(self).__name__, str(global_counter)])
        global_counter = global_counter + 1


class PyTextEncoderBase(PyTextSeq2SeqModule):
    class Config(ConfigBase):
        pass

    def forward(self):
        raise RuntimeError("Not implemented error")


class PyTextDecoderBase(PyTextSeq2SeqModule):
    class Config(ConfigBase):
        pass

    def forward(self):
        raise RuntimeError("Not implemented error")

    def get_encoder_outputs(self, encoder_out):
        raise RuntimeError("Not implemented error")

    def get_src_lengths(self, encoder_out):
        raise RuntimeError("Not implemented error")


class PyTextIncrementalDecoderComponent:
    def get_incremental_state(
        self, incremental_state: Dict[str, Tensor], key: str
    ) -> Optional[Tensor]:
        """Helper for getting incremental state for an nn.Module."""
        full_key = prepare_full_key(self.instance_id, key)

        if incremental_state is None or full_key not in incremental_state:
            return None
        return incremental_state[full_key]

    def set_incremental_state(
        self, incremental_state: Dict[str, Tensor], key: str, value
    ):
        """Helper for setting incremental state for an nn.Module."""
        if incremental_state is not None:
            full_key = prepare_full_key(self.instance_id, key)
            incremental_state[full_key] = value

    def reorder_incremental_state(
        self, incremental_state: Dict[str, Tensor], new_order: Tensor
    ):
        pass


class PyTextSeq2SeqModelBase(PyTextSeq2SeqModule):
    class Config(ConfigBase):
        pass

    def forward(self):
        raise RuntimeError("Not implemented error")


class PlaceholderIdentity(nn.Module):
    def forward(self, x, incremental_state: Optional[Dict[str, Tensor]] = None):
        return x

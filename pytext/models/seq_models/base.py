#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Dict, Optional, Tuple

from pytext.models.module import Module
from pytext.utils.usage import log_class_usage
from torch import Tensor, nn

from .utils import prepare_full_key


global_counter = 0


class PyTextSeq2SeqModule(Module):
    instance_id: str = None

    def __init__(self):
        super().__init__()
        self.assign_id()
        log_class_usage(__class__)

    def assign_id(self):
        global global_counter
        self.instance_id = ".".join([type(self).__name__, str(global_counter)])
        global_counter = global_counter + 1


class PyTextIncrementalDecoderComponent(PyTextSeq2SeqModule):
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


class PlaceholderIdentity(nn.Module):
    def forward(self, x, incremental_state: Optional[Dict[str, Tensor]] = None):
        return x


class PlaceholderAttentionIdentity(nn.Module):
    def forward(
        self,
        query,
        key,
        value,
        need_weights: bool = None,
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        optional_attention: Optional[Tensor] = None
        return query, optional_attention

    def reorder_incremental_state(
        self, incremental_state: Dict[str, Tensor], new_order: Tensor
    ):
        pass

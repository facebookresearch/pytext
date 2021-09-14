#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
from typing import Optional, List, Dict

import torch
import torch.nn.functional as F
from torch import nn, Tensor


def log_and_overwrite(param_name: str, x, y) -> int:
    if x != y:
        logging.warning(f"Mismatch of {param_name} expected {y} got {x}")
    return y


def prepare_full_key(instance_id: str, key: str, secondary_key: Optional[str] = None):
    if secondary_key is not None:
        return instance_id + "." + key + "." + secondary_key
    else:
        return instance_id + "." + key


def make_positions(input, padding_idx: int):
    """Replace non-padding symbols with their position numbers.

    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    mask = input.ne(padding_idx)
    return torch.cumsum(mask, dim=1) * mask + padding_idx


def unfold1d(x, kernel_size: int, padding_l: int, pad_value: float = 0):
    """unfold T x B x C to T x B x C x K"""
    if kernel_size > 1:
        T, B, C = x.size()
        x = F.pad(
            x, (0, 0, 0, 0, padding_l, kernel_size - 1 - padding_l), value=pad_value
        )
        x = x.as_strided((T, B, C, kernel_size), (B * C, C, 1, B * C))
    else:
        x = x.unsqueeze(3)
    return x


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


def verify_encoder_out(encoder_out: Dict[str, Tensor], keys: List[str]):
    for key in keys:
        assert key in encoder_out, f"Needed {key} to be in {encoder_out.keys()}"


def extract_ontology_vocab(target_dictionary):
    fixed_generation_vocab = []
    for i, symbol in enumerate(target_dictionary._vocab):
        lower_symbol = symbol.lower()
        if lower_symbol[0] == "[" or lower_symbol == "]":
            fixed_generation_vocab.append(i)
    return fixed_generation_vocab

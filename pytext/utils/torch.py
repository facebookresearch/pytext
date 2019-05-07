#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Dict, List

from torch import Tensor, jit


@jit.script
def list_max(l: List[int]):
    max_value = l[0]  # fine to throw if empty
    for i in range(len(l) - 1):  # don't forget the +1
        max_value = max(max_value, l[i + 1])
    return max_value


@jit.script
def list_membership(item: int, list: List[int]):
    item_present = False
    for i in list:
        if item == i:
            item_present = True
    return item_present


class Vocabulary(jit.ScriptModule):
    def __init__(
        self,
        vocab_list,
        unk_idx: int = 0,
        pad_idx: int = -1,
        bos_idx: int = -1,
        eos_idx: int = -1,
    ):
        super().__init__()
        self.vocab = jit.Attribute(vocab_list, List[str])
        self.unk_idx = jit.Attribute(unk_idx, int)
        self.pad_idx = jit.Attribute(pad_idx, int)
        self.eos_idx = jit.Attribute(eos_idx, int)
        self.bos_idx = jit.Attribute(bos_idx, int)
        self.idx = jit.Attribute(
            {word: i for i, word in enumerate(vocab_list)}, Dict[str, int]
        )

    @jit.script_method
    def lookup_indices_1d(self, values: List[str]) -> List[int]:
        result = jit.annotate(List[int], [])
        for value in values:
            result.append(self.idx.get(value, self.unk_idx))
        return result

    @jit.script_method
    def lookup_indices_2d(self, values: List[List[str]]) -> List[List[int]]:
        result = jit.annotate(List[List[int]], [])
        for value in values:
            result.append(self.lookup_indices_1d(value))
        return result

    @jit.script_method
    def lookup_words_1d(
        self, values: Tensor, filter_token_list: List[int] = ()
    ) -> List[str]:
        result = jit.annotate(List[str], [])
        for idx in range(values.size(0)):
            value = int(values[idx])
            if not list_membership(value, filter_token_list):
                if value < len(self.vocab):
                    result.append(self.vocab[int(value)])
                else:
                    result.append(self.vocab[self.unk_idx])
        return result

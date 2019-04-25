#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Dict, List

from torch import jit


@jit.script
def list_max(l: List[int]):
    max_value = l[0]  # fine to throw if empty
    for i in range(len(l) - 1):  # don't forget the +1
        max_value = max(max_value, l[i + 1])
    return max_value


class Vocabulary(jit.ScriptModule):
    def __init__(self, vocab_list, unk_idx: int = 0):
        super().__init__()
        self.vocab = jit.Attribute(vocab_list, List[str])
        self.unk_idx = jit.Attribute(unk_idx, int)
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

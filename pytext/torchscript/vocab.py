#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Dict, List, Optional

import torch
from pytext.common.constants import SpecialTokens

from .utils import list_membership


class ScriptVocabulary(torch.jit.ScriptModule):
    def __init__(
        self,
        vocab_list,
        unk_idx: int = 0,
        pad_idx: int = -1,
        bos_idx: int = -1,
        eos_idx: int = -1,
        mask_idx: int = -1,
    ):
        super().__init__()
        self.vocab = torch.jit.Attribute(vocab_list, List[str])
        self.unk_idx = torch.jit.Attribute(unk_idx, int)
        self.pad_idx = torch.jit.Attribute(pad_idx, int)
        self.eos_idx = torch.jit.Attribute(eos_idx, int)
        self.bos_idx = torch.jit.Attribute(bos_idx, int)
        self.mask_idx = torch.jit.Attribute(mask_idx, int)
        self.idx = torch.jit.Attribute(
            {word: i for i, word in enumerate(vocab_list)}, Dict[str, int]
        )
        pad_token = vocab_list[pad_idx] if pad_idx >= 0 else SpecialTokens.PAD
        self.pad_token = torch.jit.Attribute(pad_token, str)

    @torch.jit.script_method
    def lookup_indices_1d(self, values: List[str]) -> List[int]:
        result = torch.jit.annotate(List[int], [])
        for value in values:
            result.append(self.idx.get(value, self.unk_idx))
        return result

    @torch.jit.script_method
    def lookup_indices_2d(self, values: List[List[str]]) -> List[List[int]]:
        result = torch.jit.annotate(List[List[int]], [])
        for value in values:
            result.append(self.lookup_indices_1d(value))
        return result

    @torch.jit.script_method
    def lookup_words_1d(
        self,
        values: torch.Tensor,
        filter_token_list: List[int] = (),
        possible_unk_token: Optional[str] = None,
    ) -> List[str]:
        """If possible_unk_token is not None, then all UNK id's will be replaced
        by possible_unk_token instead of the default UNK string which is <UNK>.
        This is a simple way to resolve UNK's when there's a correspondence
        between source and target translations.
        """
        result = torch.jit.annotate(List[str], [])
        for idx in range(values.size(0)):
            value = int(values[idx])
            if not list_membership(value, filter_token_list):
                result.append(self.lookup_word(value, possible_unk_token))
        return result

    @torch.jit.script_method
    def lookup_words_1d_cycle_heuristic(
        self,
        values: torch.Tensor,
        filter_token_list: List[int],
        ordered_unks_token: List[str],
    ) -> List[str]:
        """This function is a extension of the possible_unk_token heuristic
        in lookup_words_1d, which fails in the case when multiple unks are
        available. The way we deal with this is we increment every unk token in
        ordered_unks_token everytime we substitute an unk token. This solves a
        substantial amount of queries with multiple unk tokens.
        """
        unk_idx = 0
        unk_idx_length = torch.jit.annotate(int, len(ordered_unks_token))
        unk_copy = torch.jit.annotate(bool, unk_idx_length != 0)
        vocab_length = torch.jit.annotate(int, len(self.vocab))

        result = torch.jit.annotate(List[str], [])
        for idx in range(values.size(0)):
            value = int(values[idx])
            if not list_membership(value, filter_token_list):
                if value < vocab_length and value != self.unk_idx:
                    result.append(self.vocab[value])
                else:
                    if not unk_copy:
                        result.append(self.vocab[self.unk_idx])
                    else:
                        unk_value = ordered_unks_token[unk_idx % unk_idx_length]
                        result.append(unk_value)
                        unk_idx += 1

        return result

    @torch.jit.script_method
    def lookup_word(self, idx: int, possible_unk_token: Optional[str] = None):
        if idx < len(self.vocab) and idx != self.unk_idx:
            return self.vocab[idx]
        else:
            return (
                self.vocab[self.unk_idx]
                if possible_unk_token is None
                else possible_unk_token
            )

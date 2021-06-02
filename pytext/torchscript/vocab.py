#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from pytext.common.constants import SpecialTokens


class ScriptVocabulary(nn.Module):
    idx: Dict[str, int]

    def __init__(
        self,
        vocab_list,
        unk_idx: int = 0,
        pad_idx: int = -1,
        bos_idx: int = -1,
        eos_idx: int = -1,
        mask_idx: int = -1,
        unk_token: Optional[str] = None,
    ):
        super().__init__()
        self.vocab: List[str] = vocab_list
        self.unk_idx: int = unk_idx
        self.pad_idx: int = pad_idx
        self.eos_idx: int = eos_idx
        self.bos_idx: int = bos_idx
        self.mask_idx: int = mask_idx
        self.idx: Dict[str, int] = {word: i for i, word in enumerate(vocab_list)}
        pad_token = vocab_list[pad_idx] if pad_idx >= 0 else SpecialTokens.PAD
        self.pad_token: str = pad_token
        self.unk_token = unk_token

    def get_pad_index(self):
        return self.pad_idx

    def get_unk_index(self):
        return self.unk_idx

    def lookup_indices_1d(self, values: List[str]) -> List[int]:
        result: List[int] = []
        for value in values:
            result.append(self.idx.get(value, self.unk_idx))
        return result

    def lookup_indices_2d(self, values: List[List[str]]) -> List[List[int]]:
        result: List[List[int]] = []
        for value in values:
            result.append(self.lookup_indices_1d(value))
        return result

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
        result: List[str] = []
        for idx in range(values.size(0)):
            value = int(values[idx])
            if not (value in filter_token_list):
                result.append(self.lookup_word(value, possible_unk_token))
        return result

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
        unk_idx_length: int = len(ordered_unks_token)
        unk_copy: bool = unk_idx_length != 0
        vocab_length: int = len(self.vocab)

        result: List[str] = []
        for idx in range(values.size(0)):
            value = int(values[idx])
            if not (value in filter_token_list):
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

    def lookup_word(self, idx: int, possible_unk_token: Optional[str] = None):
        if idx < len(self.vocab) and idx != self.unk_idx:
            return self.vocab[idx]
        else:
            return (
                self.vocab[self.unk_idx]
                if possible_unk_token is None
                else possible_unk_token
            )

    def __len__(self):
        return len(self.vocab)

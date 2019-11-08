#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List, Tuple

import torch

from .bert import ScriptBERTTensorizer


class ScriptRoBERTaTensorizer(ScriptBERTTensorizer):
    @torch.jit.script_method
    def _lookup_tokens(self, tokens: List[Tuple[str, int, int]]) -> List[int]:
        return self.vocab_lookup(
            tokens,
            bos_idx=self.vocab.bos_idx,
            eos_idx=self.vocab.eos_idx,
            use_eos_token_for_bos=False,
            max_seq_len=self.max_seq_len,
        )[0]

    @torch.jit.script_method
    def _wrap_special_tokens(self, token_ids: List[int], idx: int) -> List[int]:
        return token_ids

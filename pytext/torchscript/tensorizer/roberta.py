#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List, Tuple

import torch

from .bert import ScriptBERTTensorizer


class ScriptRoBERTaTensorizer(ScriptBERTTensorizer):
    @torch.jit.script_method
    def numberize(self, row: List[str]) -> Tuple[List[int], List[int], int]:
        tokens: List[int] = []
        segment_labels: List[int] = []
        seq_len: int = 0

        for idx, text in enumerate(row):
            token_ids: List[int] = self.vocab_lookup(
                self.tokenizer.tokenize(text),
                bos_idx=self.vocab.bos_idx,
                eos_idx=self.vocab.eos_idx,
                max_seq_len=self.max_seq_len,
            )[0]
            tokens.extend(token_ids)
            segment_labels.extend([idx] * len(token_ids))
        seq_len = len(tokens)

        return tokens, segment_labels, seq_len

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List, Optional, Tuple

import torch

from .bert import ScriptBERTTensorizer


class ScriptRoBERTaTensorizer(ScriptBERTTensorizer):
    @torch.jit.script_method
    def numberize(
        self, text_row: Optional[List[str]], token_row: Optional[List[List[str]]]
    ) -> Tuple[List[int], List[int], int]:
        token_ids: List[int] = []
        segment_labels: List[int] = []
        seq_len: int = 0
        per_sentence_tokens: List[List[Tuple[str, int, int]]] = self.tokenize(
            text_row, token_row
        )

        for idx, per_sentence_token in enumerate(per_sentence_tokens):
            lookup_ids: List[int] = self.vocab_lookup(
                per_sentence_token,
                bos_idx=self.vocab.bos_idx,
                eos_idx=self.vocab.eos_idx,
                max_seq_len=self.max_seq_len,
            )[0]
            token_ids.extend(lookup_ids)
            segment_labels.extend([idx] * len(lookup_ids))
        seq_len = len(token_ids)

        return token_ids, segment_labels, seq_len

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List, Optional, Tuple

import torch
from pytext.utils.torch import Vocabulary as ScriptVocabulary, pad_2d_mask

from .tensorizer import ScriptTensorizer, VocabLookup


class ScriptBERTTensorizer(ScriptTensorizer):
    def __init__(
        self,
        tokenizer: torch.jit.ScriptModule,
        vocab: ScriptVocabulary,
        max_seq_len: int,
        add_bos_token: bool,
        use_eos_token_for_bos: bool,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.vocab_lookup = VocabLookup(vocab)
        self.max_seq_len = torch.jit.Attribute(max_seq_len, int)
        self.add_bos_token = torch.jit.Attribute(add_bos_token, bool)
        self.use_eos_token_for_bos = torch.jit.Attribute(use_eos_token_for_bos, bool)

    @torch.jit.script_method
    def numberize(self, row: List[str]) -> Tuple[List[int], List[int], int]:
        """Convert row into token ids by doing vocab look-up. It will also
        append bos & eos index into token_ids if needed.

        Args:
            row: a list of input texts, in most case it is a
                single text or a pair of texts.

        Returns:
            a list of token ids after doing vocab lookup and segment labels.
        """
        token_ids: List[int] = []
        segment_labels: List[int] = []
        seq_len: int = 0

        for idx, text in enumerate(row):
            if idx == 0 and self.add_bos_token:
                bos_idx: Optional[int] = self.vocab.bos_idx
            else:
                bos_idx: Optional[int] = None

            lookup_ids: List[int] = self.vocab_lookup(
                self.tokenizer.tokenize(text),
                bos_idx=bos_idx,
                eos_idx=self.vocab.eos_idx,
                use_eos_token_for_bos=self.use_eos_token_for_bos,
                max_seq_len=self.max_seq_len,
            )[0]
            token_ids.extend(lookup_ids)
            segment_labels.extend([idx] * len(lookup_ids))
        seq_len = len(token_ids)

        return token_ids, segment_labels, seq_len

    @torch.jit.script_method
    def tensorize(
        self, rows: List[List[str]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens_2d: List[List[int]] = []
        segment_labels_2d: List[List[int]] = []
        seq_len_2d: List[int] = []

        for row in rows:
            numberized: Tuple[List[int], List[int], int] = self.numberize(row)
            tokens_2d.append(numberized[0])
            segment_labels_2d.append(numberized[1])
            seq_len_2d.append(numberized[2])

        tokens, pad_mask = pad_2d_mask(tokens_2d, pad_value=self.vocab.pad_idx)
        segment_labels, _ = pad_2d_mask(segment_labels_2d, pad_value=self.vocab.pad_idx)
        return tokens, pad_mask, segment_labels

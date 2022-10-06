#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List, Optional, Tuple

import torch
from pytext.torchscript.utils import pad_2d_mask

from .bert import ScriptBERTTensorizerBase


class ScriptRoBERTaTensorizer(ScriptBERTTensorizerBase):
    @torch.jit.script_method
    def _lookup_tokens(self, tokens: List[Tuple[str, int, int]]) -> List[int]:
        return self.vocab_lookup(
            tokens,
            bos_idx=self.vocab.bos_idx,
            eos_idx=self.vocab.eos_idx,
            use_eos_token_for_bos=False,
            max_seq_len=self.max_seq_len,
        )[0]


class ScriptRoBERTaTensorizerWithIndices(ScriptBERTTensorizerBase):
    @torch.jit.script_method
    def _lookup_tokens(
        self, tokens: List[Tuple[str, int, int]]
    ) -> Tuple[List[int], List[int], List[int]]:
        return self.vocab_lookup(
            tokens,
            bos_idx=self.vocab.bos_idx,
            eos_idx=self.vocab.eos_idx,
            use_eos_token_for_bos=False,
            max_seq_len=self.max_seq_len,
        )

    @torch.jit.script_method
    def numberize(
        self, text_row: Optional[List[str]], token_row: Optional[List[List[str]]]
    ) -> Tuple[List[int], List[int], List[int], List[int]]:
        token_ids: List[int] = []
        start_indices: List[int] = []
        end_indices: List[int] = []
        positions: List[int] = []
        per_sentence_tokens: List[List[Tuple[str, int, int]]] = self.tokenize(
            text_row, token_row
        )

        for idx, per_sentence_token in enumerate(per_sentence_tokens):
            lookup_ids, start_ids, end_ids = self._lookup_tokens(per_sentence_token)
            lookup_ids = self._wrap_numberized_tokens(lookup_ids, idx)
            token_ids.extend(lookup_ids)
            start_indices.extend(start_ids)
            end_indices.extend(end_ids)

        seq_len = len(token_ids)
        positions = list(range(seq_len))

        return token_ids, start_indices, end_indices, positions

    @torch.jit.script_method
    def tensorize(
        self,
        texts: Optional[List[List[str]]] = None,
        tokens: Optional[List[List[List[str]]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens_2d: List[List[int]] = []
        start_indices_2d: List[List[int]] = []
        end_indices_2d: List[List[int]] = []
        positions_2d: List[List[int]] = []

        for idx in range(self.batch_size(texts, tokens)):
            numberized: Tuple[
                List[int], List[int], List[int], List[int]
            ] = self.numberize(
                self.get_texts_by_index(texts, idx),
                self.get_tokens_by_index(tokens, idx),
            )
            tokens_2d.append(numberized[0])
            start_indices_2d.append(numberized[1])
            end_indices_2d.append(numberized[2])
            positions_2d.append(numberized[3])

        tokens, pad_mask = pad_2d_mask(
            tokens_2d,
            pad_value=self.vocab.pad_idx,
            seq_padding_control=self.seq_padding_control,
            max_seq_pad_len=self.max_seq_len,
            batch_padding_control=self.batch_padding_control,
        )

        start_indices, _ = pad_2d_mask(
            start_indices_2d,
            pad_value=0,
            seq_padding_control=self.seq_padding_control,
            max_seq_pad_len=self.max_seq_len,
            batch_padding_control=self.batch_padding_control,
        )

        end_indices, _ = pad_2d_mask(
            end_indices_2d,
            pad_value=0,
            seq_padding_control=self.seq_padding_control,
            max_seq_pad_len=self.max_seq_len,
            batch_padding_control=self.batch_padding_control,
        )

        positions, _ = pad_2d_mask(
            positions_2d,
            pad_value=0,
            seq_padding_control=self.seq_padding_control,
            max_seq_pad_len=self.max_seq_len,
            batch_padding_control=self.batch_padding_control,
        )

        if self.device == "":
            return tokens, pad_mask, start_indices, end_indices, positions
        else:
            return (
                tokens.to(self.device),
                pad_mask.to(self.device),
                start_indices.to(self.device),
                end_indices.to(self.device),
                positions.to(self.device),
            )

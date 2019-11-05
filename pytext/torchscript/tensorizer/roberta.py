#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List, Tuple

import torch

from .bert import ScriptBERTTensorizerBase


class ScriptRoBERTaTensorizerBase(ScriptBERTTensorizerBase):
    @torch.jit.script_method
    def numberize(self, row: List[str]) -> Tuple[List[int], List[int], int]:
        """Convert raw inputs into token ids by doing vocab look-up. It will also
        append bos & eos index into token ids if needed.

        Args:
            row: 1) a list of raw inputs, in most case it is a
                single text or a pair of texts.
                 2) a list of preprocced tokens, we could still
                apply other operations (for example: bpe) on it.

        Returns:
            a list of token ids after doing vocab lookup and segment labels.
        """
        token_ids: List[int] = []
        segment_labels: List[int] = []
        seq_len: int = 0
        per_sentence_tokens: List[List[Tuple[str, int, int]]] = self.tokenize(row)

        for idx, tokens in enumerate(per_sentence_tokens):
            lookup_ids: List[int] = self.vocab_lookup(
                tokens,
                bos_idx=self.vocab.bos_idx,
                eos_idx=self.vocab.eos_idx,
                max_seq_len=self.max_seq_len,
            )[0]
            token_ids.extend(lookup_ids)
            segment_labels.extend([idx] * len(lookup_ids))
        seq_len = len(token_ids)

        return token_ids, segment_labels, seq_len


class ScriptRoBERTaTensorizer(ScriptRoBERTaTensorizerBase):
    @torch.jit.script_method
    def tokenize(self, row: List[str]) -> List[List[Tuple[str, int, int]]]:
        """Convert raw inputs into tokens.

        Args:
            row: a list of raw inputs, in most case it is a
                single text or a pair of texts.

        Returns:
            a per sentence list of tokens which include token index.
        """

        per_sentence_tokens: List[List[Tuple[str, int, int]]] = []
        for text in row:
            per_sentence_tokens.append(self.tokenizer.tokenize(text))
        return per_sentence_tokens


class ScriptRoBERTaTokenTensorizer(ScriptRoBERTaTensorizerBase):
    @torch.jit.script_method
    def tokenize(self, row: List[str]) -> List[List[Tuple[str, int, int]]]:
        """Convert raw inputs into tokens.

        Args:
            row: a list of raw inputs, in most case it is a
                single text or a pair of texts.

        Returns:
            a per sentence list of tokens which include token index.
        """

        per_sentence_tokens: List[Tuple[str, int, int]] = []
        for raw_token in row:
            per_sentence_tokens.extend(self.tokenizer.tokenize(raw_token))
        return [per_sentence_tokens]

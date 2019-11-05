#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List, Optional, Tuple

import torch
from pytext.torchscript.utils import pad_2d_mask
from pytext.torchscript.vocab import ScriptVocabulary

from .tensorizer import ScriptTensorizer, VocabLookup


class ScriptBERTTensorizerBase(ScriptTensorizer):
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
            if idx == 0 and self.add_bos_token:
                bos_idx: Optional[int] = self.vocab.bos_idx
            else:
                bos_idx: Optional[int] = None

            lookup_ids: List[int] = self.vocab_lookup(
                tokens,
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
        """Convert multiple rows of raw inputs into model input tensors.

        Args:
            row: 1) each row is a list of raw inputs, in most case it is a
                single text or a pair of texts.
                 2) each row is a list of preprocced tokens, we could still
                apply other operations (for example: bpe) on it.

        Returns:
            model input tensors.
        """

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


class ScriptBERTTensorizer(ScriptBERTTensorizerBase):
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


class ScriptBERTTokenTensorizer(ScriptBERTTensorizerBase):
    @torch.jit.script_method
    def tokenize(self, row: List[str]) -> List[List[Tuple[str, int, int]]]:
        """Convert raw inputs into tokens.

        Args:
            row: a list of preprocced tokens, we could still
                apply other operations (for example: bpe) on it.

        Returns:
            a per sentence list of tokens which include token index.
        """

        per_sentence_tokens: List[Tuple[str, int, int]] = []
        for raw_token in row:
            per_sentence_tokens.extend(self.tokenizer.tokenize(raw_token))
        return [per_sentence_tokens]

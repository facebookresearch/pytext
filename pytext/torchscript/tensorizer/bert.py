#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List, Optional, Tuple

import torch
from pytext.torchscript.utils import pad_2d, pad_2d_mask
from pytext.torchscript.vocab import ScriptVocabulary

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
    def tokenize(
        self, text_row: Optional[List[str]], token_row: Optional[List[List[str]]]
    ) -> List[List[Tuple[str, int, int]]]:
        per_sentence_tokens: List[List[Tuple[str, int, int]]] = []

        if text_row is not None:
            for text in text_row:
                per_sentence_tokens.append(self.tokenizer.tokenize(text))
        elif token_row is not None:
            for sentence_raw_tokens in token_row:
                sentence_tokens: List[Tuple[str, int, int]] = []
                for raw_token in sentence_raw_tokens:
                    sentence_tokens.extend(self.tokenizer.tokenize(raw_token))
                per_sentence_tokens.append(sentence_tokens)

        return per_sentence_tokens

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
            if idx == 0 and self.add_bos_token:
                bos_idx: Optional[int] = self.vocab.bos_idx
            else:
                bos_idx: Optional[int] = None

            lookup_ids: List[int] = self.vocab_lookup(
                per_sentence_token,
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
        self,
        texts: Optional[List[List[str]]] = None,
        tokens: Optional[List[List[List[str]]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens_2d: List[List[int]] = []
        segment_labels_2d: List[List[int]] = []
        seq_len_2d: List[int] = []

        for idx in range(self.batch_size(texts, tokens)):
            numberized: Tuple[List[int], List[int], int] = self.numberize(
                self.get_texts_by_index(texts, idx),
                self.get_tokens_by_index(tokens, idx),
            )
            tokens_2d.append(numberized[0])
            segment_labels_2d.append(numberized[1])
            seq_len_2d.append(numberized[2])

        tokens, pad_mask = pad_2d_mask(tokens_2d, pad_value=self.vocab.pad_idx)
        segment_labels = torch.tensor(
            pad_2d(segment_labels_2d, seq_lens=seq_len_2d, pad_idx=self.vocab.pad_idx),
            dtype=torch.long,
        )

        return tokens, pad_mask, segment_labels

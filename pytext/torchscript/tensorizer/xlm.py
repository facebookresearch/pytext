#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List, Optional, Tuple

import torch
from pytext.torchscript.utils import pad_2d, pad_2d_mask
from pytext.torchscript.vocab import ScriptVocabulary

from .tensorizer import ScriptTensorizer, VocabLookup


class ScriptXLMTensorizer(ScriptTensorizer):
    def __init__(
        self,
        tokenizer: torch.jit.ScriptModule,
        token_vocab: ScriptVocabulary,
        language_vocab: ScriptVocabulary,
        max_seq_len: int,
        default_language: str,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.token_vocab = token_vocab
        self.language_vocab = language_vocab
        self.token_vocab_lookup = VocabLookup(token_vocab)
        self.language_vocab_lookup = VocabLookup(language_vocab)

        self.max_seq_len = torch.jit.Attribute(max_seq_len, int)
        self.default_language = torch.jit.Attribute(default_language, str)

    @torch.jit.script_method
    def tokenize(
        self,
        text_row: Optional[List[str]],
        token_row: Optional[List[List[str]]],
        language_row: List[str],
    ) -> Tuple[List[List[Tuple[str, int, int]]], List[List[Tuple[str, int, int]]]]:
        per_sentence_tokens: List[List[Tuple[str, int, int]]] = []
        per_sentence_languages: List[List[Tuple[str, int, int]]] = []

        if text_row is not None:
            """
            Tokenize every single text into a list of tokens.
            For example:
            text_row = ["hello world", "this is sentence"]
            per_sentence_tokens = [["hello", "world"], ["this", "is", "sentence"]]
            """
            for idx, text in enumerate(text_row):
                sentence_tokens: List[Tuple[str, int, int]] = self.tokenizer.tokenize(
                    text
                )
                sentence_languages: List[Tuple[str, int, int]] = [
                    (language_row[idx], token[1], token[2]) for token in sentence_tokens
                ]

                per_sentence_tokens.append(sentence_tokens)
                per_sentence_languages.append(sentence_languages)
        elif token_row is not None:
            """
            Tokenize every single token into a sub tokens. (example: BPE)
            For example:
            token_row = [["hello", "world"], ["this", "is", "sentence"]]
            per_sentence_tokens = [
                ["he", "llo" "wo", "rld"], ["th", "is", "is", "sen", "tence"]
            ]
            """
            for idx, sentence_raw_tokens in enumerate(token_row):
                sentence_tokens: List[Tuple[str, int, int]] = []
                sentence_languages: List[Tuple[str, int, int]] = []

                for raw_token in sentence_raw_tokens:
                    sub_tokens: List[Tuple[str, int, int]] = self.tokenizer.tokenize(
                        raw_token
                    )
                    sub_languages: List[Tuple[str, int, int]] = [
                        (language_row[idx], token[1], token[2]) for token in sub_tokens
                    ]

                    sentence_tokens.extend(sub_tokens)
                    sentence_languages.extend(sub_languages)

                per_sentence_tokens.append(sentence_tokens)
                per_sentence_languages.append(sentence_languages)

        return per_sentence_tokens, per_sentence_languages

    @torch.jit.script_method
    def _lookup_tokens(
        self,
        tokens: List[Tuple[str, int, int]],
        languages: List[Tuple[str, int, int]],
        max_seq_len: int,
    ) -> Tuple[List[int], List[int]]:
        token_ids: List[int] = self.token_vocab_lookup(
            tokens,
            bos_idx=self.token_vocab.eos_idx,
            eos_idx=self.token_vocab.eos_idx,
            use_eos_token_for_bos=True,
            max_seq_len=max_seq_len,
        )[0]
        language_special_idx: int = self.language_vocab.idx.get(
            languages[0][0], self.language_vocab.unk_idx
        )
        language_ids = self.language_vocab_lookup(
            languages,
            bos_idx=language_special_idx,
            eos_idx=language_special_idx,
            use_eos_token_for_bos=True,
            max_seq_len=max_seq_len,
        )[0]
        return token_ids, language_ids

    @torch.jit.script_method
    def numberize(
        self,
        text_row: Optional[List[str]],
        token_row: Optional[List[List[str]]],
        language_row: List[str],
    ) -> Tuple[List[int], List[int], int, List[int]]:
        per_sentence_tokens, per_sentence_languages = self.tokenize(
            text_row, token_row, language_row
        )

        token_ids: List[int] = []
        language_ids: List[int] = []
        max_seq_len: int = self.max_seq_len // len(per_sentence_tokens)

        # concatinate tokens from each text in the same row into a single list of tokens
        for idx in range(len(per_sentence_tokens)):
            lookup_token_ids, lookup_language_ids = self._lookup_tokens(
                per_sentence_tokens[idx], per_sentence_languages[idx], max_seq_len
            )
            token_ids.extend(lookup_token_ids)
            language_ids.extend(lookup_language_ids)
        seq_len: int = len(token_ids)
        positions: List[int] = [i for i in range(seq_len)]

        return token_ids, language_ids, seq_len, positions

    @torch.jit.script_method
    def tensorize(
        self,
        texts: Optional[List[List[str]]] = None,
        tokens: Optional[List[List[List[str]]]] = None,
        languages: Optional[List[List[str]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # unwrap Optional
        batch_size: int = self.batch_size(texts, tokens)
        row_size: int = self.row_size(texts, tokens)

        if languages is None:
            languages = [[self.default_language] * row_size] * batch_size

        tokens_2d: List[List[int]] = []
        languages_2d: List[List[int]] = []
        seq_len_2d: List[int] = []
        positions_2d: List[List[int]] = []

        for idx in range(batch_size):
            numberized: Tuple[List[int], List[int], int, List[int]] = self.numberize(
                self.get_texts_by_index(texts, idx),
                self.get_tokens_by_index(tokens, idx),
                languages[idx],
            )
            tokens_2d.append(numberized[0])
            languages_2d.append(numberized[1])
            seq_len_2d.append(numberized[2])
            positions_2d.append(numberized[3])

        tokens, pad_mask = pad_2d_mask(tokens_2d, pad_value=self.token_vocab.pad_idx)
        languages = torch.tensor(
            pad_2d(languages_2d, seq_lens=seq_len_2d, pad_idx=0), dtype=torch.long
        )
        positions = torch.tensor(
            pad_2d(positions_2d, seq_lens=seq_len_2d, pad_idx=0), dtype=torch.long
        )

        if self.device == "":
            return tokens, pad_mask, languages, positions
        else:
            return (
                tokens.to(self.device),
                pad_mask.to(self.device),
                languages.to(self.device),
                positions.to(self.device),
            )

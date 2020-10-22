#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List, Tuple

import torch

from .bpe import ScriptBPE


class ScriptTokenizerBase(torch.jit.ScriptModule):
    @torch.jit.script_method
    def tokenize(self, input: str) -> List[Tuple[str, int, int]]:
        """
        Process a single line of raw inputs into tokens, it supports
        two input formats:
        1) a single text
        2) a token

        Returns a list of tokens with start and end indices in original input.
        """
        raise NotImplementedError


class ScriptDoNothingTokenizer(ScriptTokenizerBase):
    @torch.jit.script_method
    def tokenize(self, raw_token: str) -> List[Tuple[str, int, int]]:
        return [(raw_token, -1, -1)]


class ScriptBPETokenizer(ScriptTokenizerBase):
    def __init__(self, bpe: ScriptBPE):
        super().__init__()
        self.bpe = bpe

    @torch.jit.script_method
    def tokenize(self, raw_token: str) -> List[Tuple[str, int, int]]:
        tokens = torch.jit.annotate(List[Tuple[str, int, int]], [])

        for bpe_token in self.bpe.bpe_token(raw_token):
            tokens.append((bpe_token, -1, -1))

        return tokens


class ScriptWordTokenizer(ScriptTokenizerBase):
    def __init__(self, lowercase=True):
        super().__init__()
        self.lowercase = lowercase

    @torch.jit.script_method
    def tokenize(self, raw_token: str) -> List[Tuple[str, int, int]]:
        """
        This tokenizers splits a raw_token into its constituent words by splitting
        the raw_token on space. This function handle multiple spaces between
        words too.
        Note:
        torch scripting doesn't support try-except and since re.finditer uses
        try in its implemetation regex based tokenization is not supported.
        """
        tokenize_input = raw_token.lower() if self.lowercase else raw_token
        tokens = tokenize_input.split()
        torchify_tokens = torch.jit.annotate(List[Tuple[str, int, int]], [])
        start, end = 0, 0
        for token in tokens:
            start = tokenize_input.find(token, end, -1)
            end = start + len(token)
            torchify_tokens.append((token.strip(), start, end))
            start = end + 1
        return torchify_tokens

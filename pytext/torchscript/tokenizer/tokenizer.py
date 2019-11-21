#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List, Tuple

import torch
from pytext.torchscript.utils import ScriptInputType

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

    def input_type(self) -> ScriptInputType:
        """
        Determine TorchScript module input type, currently it have four types
        1) text: batch with a single text in each row, List[str]
        2) tokens: batch with a list of tokens from single text
        in each row, List[List[str]]
        3) multi_text: batch with multiple texts in each row,
        List[List[str]]
        4) multi_tokens: batch with multiple lists of tokens from
        multiple texts in each row, List[List[List[str]]]
        """
        raise NotImplementedError


class ScriptTextTokenizerBase(ScriptTokenizerBase):
    def input_type(self) -> ScriptInputType:
        return ScriptInputType.text


class ScriptTokenTokenizerBase(ScriptTokenizerBase):
    def input_type(self) -> ScriptInputType:
        return ScriptInputType.token


class ScriptDoNothingTokenizer(ScriptTokenTokenizerBase):
    @torch.jit.script_method
    def tokenize(self, raw_token: str) -> List[Tuple[str, int, int]]:
        return [(raw_token, -1, -1)]


class ScriptBPETokenizer(ScriptTokenTokenizerBase):
    def __init__(self, bpe: ScriptBPE):
        super().__init__()
        self.bpe = bpe

    @torch.jit.script_method
    def tokenize(self, raw_token: str) -> List[Tuple[str, int, int]]:
        tokens = torch.jit.annotate(List[Tuple[str, int, int]], [])

        for bpe_token in self.bpe.bpe_token(raw_token):
            tokens.append((bpe_token, -1, -1))

        return tokens

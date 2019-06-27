#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import re
from typing import List, NamedTuple

from pytext.config import ConfigBase
from pytext.config.component import Component, ComponentType, create_component
from pytorch_pretrained_bert.tokenization import (
    BasicTokenizer,
    WordpieceTokenizer,
    load_vocab,
)


class Token(NamedTuple):
    value: str
    start: int
    end: int


class Tokenizer(Component):
    """A simple regex-splitting tokenizer."""

    __COMPONENT_TYPE__ = ComponentType.TOKENIZER
    __EXPANSIBLE__ = True

    class Config(Component.Config):
        #: A regular expression for the tokenizer to split on. Tokens are the segments
        #: between the regular expression matches. The start index is inclusive of the
        #: unmatched region, and the end index is exclusive (matching the first
        #: character of the matched split region).
        split_regex: str = r"\s+"
        #: Whether token values should be lowercased or not.
        lowercase: bool = True

    @classmethod
    def from_config(cls, config: Config):
        return cls(config.split_regex, config.lowercase)

    def __init__(self, split_regex=r"\s+", lowercase=True):
        super().__init__(None)
        self.split_regex = split_regex
        self.lowercase = lowercase

    def tokenize(self, input: str) -> List[Token]:
        tokens = []
        start = 0
        tokenize_input = input.lower() if self.lowercase else input
        for match in re.finditer(self.split_regex, tokenize_input):
            split_start, split_end = match.span()
            tokens.append(Token(tokenize_input[start:split_start], start, split_start))
            start = split_end
        tokens.append(Token(tokenize_input[start : len(input)], start, len(input)))
        return [token for token in tokens if token.value]


class DoNothingTokenizer(Tokenizer):
    """
    Tokenizer that takes a list of strings and converts to a list of Tokens.
    Useful in cases where tokenizer is run before-hand
    """

    class Config(Component.Config):
        do_nothing: str = ""

    @classmethod
    def from_config(cls, config: Config):
        return cls()

    def __init__(self):
        super().__init__(None)

    def tokenize(self, input: List[str]) -> List[Token]:
        tokens = [Token(token_text, -1, -1) for token_text in input if token_text]
        return tokens


class BERTInitialTokenizer(Tokenizer):
    """
    Basic initial tokenization for BERT.  This is run prior to word piece, does
    white space tokenization in addition to lower-casing and accent removal
    if specified.
    """

    class Config(Tokenizer.Config):
        """Config for this class."""

    @classmethod
    def from_config(cls, config: Config):
        basic_tokenizer = BasicTokenizer(do_lower_case=config.lowercase)
        return cls(basic_tokenizer)

    def __init__(self, basic_tokenizer) -> None:
        self.tokenizer = basic_tokenizer

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        if self.tokenizer.do_lower_case:
            text = self.tokenizer._run_strip_accents(text.lower())
        tokens = self.tokenizer.tokenize(text)
        end = 0
        result = []
        for token in tokens:
            start = text.find(token, end)
            if start == -1:  # safety check, this should not happen
                start = end
            end = start + len(token)
            result.append(Token(token, start, end))
        return result


class WordPieceTokenizer(Tokenizer):
    """Word piece tokenizer for BERT models."""

    class Config(ConfigBase):
        basic_tokenizer: BERTInitialTokenizer.Config = BERTInitialTokenizer.Config()
        wordpiece_vocab_path: str = "/mnt/vol/nlp_technologies/bert/uncased_L-12_H-768_A-12/vocab.txt"

    def __init__(self, wordpiece_vocab, basic_tokenizer, wordpiece_tokenizer) -> None:
        self.vocab = wordpiece_vocab
        self.basic_tokenizer = basic_tokenizer
        self.wordpiece_tokenizer = wordpiece_tokenizer

    @classmethod
    def from_config(cls, config: Config):
        basic_tokenizer = create_component(
            ComponentType.TOKENIZER, config.basic_tokenizer
        )
        vocab = load_vocab(config.wordpiece_vocab_path)
        wordpiece_tokenizer = WordpieceTokenizer(vocab=vocab)
        return cls(vocab, basic_tokenizer, wordpiece_tokenizer)

    def tokenize(self, input_str: str) -> List[Token]:
        tokens = []
        for token in self.basic_tokenizer.tokenize(input_str):
            start = token.start
            for sub_token in self.wordpiece_tokenizer.tokenize(token.value):
                piece_len = (
                    len(sub_token)
                    if not sub_token.startswith("##")
                    else (len(sub_token) - 2)  # account for ##
                )
                end = start + piece_len
                tokens.append(Token(sub_token, start, end))
                start = end
        return [token for token in tokens if token.value]

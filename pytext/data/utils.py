#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import collections
import itertools
import re
from typing import List, NamedTuple, Tuple

import torch
from pytext.config.component import Component, ComponentType


class Token(NamedTuple):
    value: str
    start: int
    end: int


class Tokenizer(Component):
    """A simple regex-splitting tokenizer."""

    __COMPONENT_TYPE__ = ComponentType.TOKENIZER

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


def should_iter(i):
    """Whether or not an object looks like a python iterable (not including strings)."""
    return (
        hasattr(i, "__iter__")
        and not isinstance(i, str)
        and not (isinstance(i, torch.Tensor) and len(i) == 0)
    )


def _infer_pad_shape(nested_lists):
    """Return the minimal tensor shape which could contain the input data."""
    yield len(nested_lists)
    while nested_lists and all(should_iter(i) for i in nested_lists):
        yield max(len(nested) for nested in nested_lists)
        nested_lists = list(itertools.chain.from_iterable(nested_lists))


def _make_nested_padding(pad_shape, pad_token):
    """Create nested lists of pad_token of shape pad_shape."""
    result = [pad_token]
    for dimension in reversed(pad_shape):
        result = [result * dimension]
    return result[0]


def pad(nested_lists, pad_token, pad_shape=None):
    """Pad the input lists with the pad token. If pad_shape is provided, pad to that
    shape, otherwise infer the input shape and pad out to a square tensor shape."""
    if pad_shape is None:
        pad_shape = list(_infer_pad_shape(nested_lists))
    if not pad_shape:
        return nested_lists
    dimension, *rest = pad_shape
    result = [pad(nested, pad_token, rest) for nested in nested_lists]
    result += [_make_nested_padding(rest, pad_token)] * (dimension - len(result))
    return result


def pad_and_tensorize(batch, pad_token=0, dtype=torch.long):
    batch = list(batch)
    if not batch:
        return torch.Tensor()

    return torch.tensor(pad(batch, pad_token=pad_token), dtype=dtype)


class SpecialToken(str):
    def __eq__(self, other):
        # We don't want to compare as equal to actual strings, but we want to behave
        # like a string code-wise.
        return self is other

    __hash__ = str.__hash__


UNK = SpecialToken("__UNKNOWN__")
PAD = SpecialToken("__PAD__")


class Vocabulary:
    """A mapping from indices to vocab elements."""

    def __init__(self, vocab_list, counts=None):
        self._vocab = vocab_list
        self.counts = counts
        self.idx = {word: i for i, word in enumerate(vocab_list)}

    def lookup_all(self, nested_values):
        """
        Look up a value or nested container of values in the vocab index.
        The return value will have the same shape as the input, with all values
        replaced with their respective indicies.
        """
        if UNK in self.idx:
            unk_idx = self.idx[UNK]
            lookup = lambda value: self.idx.get(value, unk_idx)
        else:
            lookup = self.idx.__getitem__

        def lookup_value(value):
            return self.lookup_all(value) if should_iter(value) else lookup(value)

        if not should_iter(nested_values):
            return lookup_value(nested_values)
        else:
            return [lookup_value(value) for value in nested_values]

    def __getitem__(self, item):
        return self._vocab[item]

    def __len__(self):
        return len(self._vocab)


class VocabBuilder:
    """Helper class for aggregating and building `Vocabulary` objects."""

    def __init__(self):
        self._counter = collections.Counter()
        self.use_unk = True
        self.unk_index = 0
        self.use_pad = True
        self.pad_index = 1

    def add_all(self, nested_values) -> None:
        """Count a value or nested container of values in the vocabulary."""
        for value in nested_values:
            if should_iter(value):
                self.add_all(value)
            else:
                self.add(value)

    def add(self, value) -> None:
        """Count a single value in the vocabulary."""
        self._counter[value] += 1

    def make_vocab(self) -> Vocabulary:
        """Build a Vocabulary object from the values seen by the builder."""
        vocab_list = list(self._counter)

        tokens_to_insert: List[Tuple[int, object]] = []
        if self.use_unk:
            tokens_to_insert.append((self.unk_index, UNK))
        if self.use_pad:
            tokens_to_insert.append((self.pad_index, PAD))
        for index, token in sorted(tokens_to_insert):
            vocab_list.insert(index, token)

        return Vocabulary(vocab_list, counts=self._counter)

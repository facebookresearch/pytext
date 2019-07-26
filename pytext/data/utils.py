#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import collections
import itertools
import re
from typing import Dict, List, Tuple

import torch
from pytext.config.component import Component, ComponentType
from pytext.utils import cuda, precision


def should_iter(i):
    """Whether or not an object looks like a python iterable (not including strings)."""
    return (
        hasattr(i, "__iter__")
        and not isinstance(i, str)
        and not (isinstance(i, torch.Tensor) and (i.dim() == 0 or len(i) == 0))
    )


def _infer_pad_shape(nested_lists):
    """Return the minimal tensor shape which could contain the input data."""
    yield len(nested_lists)
    while nested_lists and all(should_iter(i) for i in nested_lists):
        # pad shape to be multiple of 8 when fp16 enabled
        yield precision.pad_length(max(len(nested) for nested in nested_lists))
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


def pad_and_tensorize(batch, pad_token=0, pad_shape=None, dtype=torch.long):
    batch = list(batch)
    if not batch:
        return torch.Tensor()

    return cuda.tensor(
        pad(batch, pad_token=pad_token, pad_shape=pad_shape), dtype=dtype
    )


def shard(rows, rank, num_workers):
    """Only return every num_workers example for distributed training."""
    queue = []
    for row in rows:
        queue.append(row)
        # might discard remainder %num_workers rows because distributed
        # training needs to be in sync
        if len(queue) == num_workers:
            yield queue[rank]
            queue = []


class SpecialToken(str):
    def __eq__(self, other):
        # We don't want to compare as equal to actual strings, but we want to behave
        # like a string code-wise. Don't use `is` comparison because we want
        # SpecialToken instances created across picklings to equals-compare
        return isinstance(other, SpecialToken) and super().__eq__(other)

    __hash__ = str.__hash__


UNK = SpecialToken("__UNKNOWN__")
PAD = SpecialToken("__PAD__")
BOS = SpecialToken("__BEGIN_OF_SENTENCE__")
EOS = SpecialToken("__END_OF_SENTENCE__")
BOL = SpecialToken("__BEGIN_OF_LIST__")
EOL = SpecialToken("__END_OF_LIST__")
MASK = SpecialToken("__MASK__")


class Vocabulary:
    """A mapping from indices to vocab elements."""

    def __init__(self, vocab_list, counts=None, replacements=None, pad_token=PAD):
        self._vocab = vocab_list
        self.counts = counts
        self.idx = {word: i for i, word in enumerate(vocab_list)}
        self.pad_token = pad_token
        if replacements:
            self.replace_tokens(replacements)
        self.unk_token_counter = [0, 0]  # count of unk tokens, total tokens
        # count of examples with least 75% unk tokens, total examples
        self.unk_example_counter = [0, 0]
        self.messages_printed = 0

    def replace_tokens(self, replacements):
        """Replace tokens in vocab with given replacement.
           Used for replacing special strings for special tokens.
           e.g. '[UNK]' for UNK"""
        for token, replacement in replacements.items():
            idx = self.idx.pop(token)
            self._vocab[idx] = replacement
            self.idx[replacement] = idx

    def lookup_all(self, nested_values):
        res, unk_counter, total = self.lookup_all_internal(nested_values)
        self.unk_token_counter[0] += unk_counter
        self.unk_token_counter[1] += total
        self.unk_example_counter[1] += 1
        if total > 3 and (unk_counter / total) > 0.75:
            self.unk_example_counter[0] += 1
            if self.unk_example_counter[0] % 100 == 0 and self.messages_printed < 200:
                self.messages_printed += 1
                c1, c2 = self.unk_token_counter
                print("")
                print(f"{c1} out of {c2} ({int(100 * c1 / c2)}%) tokens not in vocab")
                c1, c2 = self.unk_example_counter
                print(
                    f"{c1} out of {c2} ({int(100 * c1 / c2)}%) examples have >= 75% "
                    f"tokens not in vocab"
                )
                print("Example: (first 20 tokens)")
                print(nested_values[:20], flush=True)
        return res

    def lookup_all_internal(self, nested_values):
        """
        Look up a value or nested container of values in the vocab index.
        The return value will have the same shape as the input, with all values
        replaced with their respective indicies.
        """

        def lookup(value):
            if UNK in self.idx:
                unk_idx = self.idx[UNK]
                v = self.idx.get(value, unk_idx)
                return v, 1 if v == unk_idx else 0, 1
            else:
                return self.idx.__getitem__(value), 0, 1

        def lookup_value(value):
            return (
                self.lookup_all_internal(value) if should_iter(value) else lookup(value)
            )

        if not should_iter(nested_values):
            return lookup_value(nested_values)
        else:
            indices = []
            unks = 0
            total = 0
            for value in nested_values:
                v, unk, t = lookup_value(value)
                indices.append(v)
                unks += unk
                total += t
            return indices, unks, total

    def get_pad_index(self):
        return self.idx[self.pad_token]

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
        # Some tokenization libraries use special pad tokens, expose this so it
        # can be configured
        self.pad_token = PAD
        self.use_bos = False
        self.bos_index = 2
        self.use_eos = False
        self.eos_index = 3
        self.use_bol = False
        self.bol_index = 4
        self.use_eol = False
        self.eol_index = 5

    def add_all(self, values) -> None:
        """Count a value or nested container of values in the vocabulary."""
        if should_iter(values):
            for value in values:
                self.add_all(value)
        else:
            self.add(values)

    def add(self, value) -> None:
        """Count a single value in the vocabulary."""
        self._counter[value] += 1

    def add_from_file(self, file_pointer, skip_header_line, lowercase_tokens, size):
        vocab_from_file = set()
        if skip_header_line:
            next(file_pointer)
        for i, line in enumerate(file_pointer):
            if size and len(vocab_from_file) == size:
                print(
                    f"Read {i + 1} items from vocab file and loaded {size} tokens. "
                    f"Skipping rest of the file."
                )
                break
            token = line.split()[0].strip()
            if lowercase_tokens:
                token = token.lower()
            vocab_from_file.add(token)
        self.add_all(sorted(vocab_from_file))

    def make_vocab(self) -> Vocabulary:
        """Build a Vocabulary object from the values seen by the builder."""
        tokens_to_insert: List[Tuple[int, object]] = []
        if self.use_unk:
            tokens_to_insert.append((self.unk_index, UNK))
            del self._counter[UNK]
        if self.use_pad:
            tokens_to_insert.append((self.pad_index, self.pad_token))
            del self._counter[self.pad_token]
        if self.use_bos:
            tokens_to_insert.append((self.bos_index, BOS))
            del self._counter[BOS]
        if self.use_eos:
            tokens_to_insert.append((self.eos_index, EOS))
            del self._counter[EOS]
        if self.use_bol:
            tokens_to_insert.append((self.bol_index, BOL))
            del self._counter[BOL]
        if self.use_eol:
            tokens_to_insert.append((self.eol_index, EOL))
            del self._counter[EOL]
        vocab_list = list(self._counter)
        for index, token in sorted(tokens_to_insert):
            vocab_list.insert(index, token)

        return Vocabulary(vocab_list, counts=self._counter, pad_token=self.pad_token)

    def truncate_to_vocab_size(self, vocab_size) -> None:
        if len(self._counter) > vocab_size > 0:
            self._counter = collections.Counter(
                dict(self._counter.most_common(vocab_size))
            )


def align_target_labels(
    targets_list: List[List[float]],
    labels_list: List[List[str]],
    label_vocab: Dict[str, int],
) -> List[List[float]]:
    """
    Given `targets_list` that are ordered according to `labels_list`, align the targets
    to match the order of `label_vocab`.
    """
    return [
        align_target_label(targets, labels, label_vocab)
        for targets, labels in zip(targets_list, labels_list)
    ]


def align_target_label(
    targets: List[float], labels: List[str], label_vocab: Dict[str, int]
) -> List[float]:
    """
    Given `targets` that are ordered according to `labels`, align the targets to match
    the order of `label_vocab`.
    """
    assert sorted(labels) == sorted(label_vocab)
    assert len(targets) == len(labels)
    aligned_targets = [None] * len(targets)
    for target, label in zip(targets, labels):
        aligned_targets[label_vocab[label]] = target
    assert all(t is not None for t in aligned_targets)
    return aligned_targets

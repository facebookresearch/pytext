#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import json
import re
from typing import List, Optional, Tuple, Type

import torch
from pytext.common import Padding
from pytext.config.component import Component, ComponentType, create_component
from pytext.data.sources.data_source import Gazetteer
from pytext.data.tokenizers import Token, Tokenizer
from pytext.utils.data import Slot

from .utils import (
    BOS,
    EOS,
    PAD,
    SpecialToken,
    VocabBuilder,
    align_target_label,
    pad_and_tensorize,
)


class Tensorizer(Component):
    """Tensorizers are a component that converts from batches of
    `pytext.data.type.DataType` instances to tensors. These tensors will eventually
    be inputs to the model, but the model is aware of the tensorizers and can arrange
    the tensors they create to conform to its model.

    Tensorizers have an initialize function. This function allows the tensorizer to
    read through the training dataset to build up any data that it needs for
    creating the model. Commonly this is valuable for things like inferring a
    vocabulary from the training set, or learning the entire set of training labels,
    or slot labels, etc.
    """

    __COMPONENT_TYPE__ = ComponentType.TENSORIZER
    __EXPANSIBLE__ = True

    class Config(Component.Config):
        pass

    def __init__(self, column_schema: List[Tuple[str, Type]]):
        self.column_schema = column_schema

    def numberize(self, row):
        raise NotImplementedError

    def sort_key(self, row):
        raise NotImplementedError

    def tensorize(self, batch):
        """Tensorizer knows how to pad and tensorize a batch of it's own output."""
        return batch

    def initialize(self):
        """
        The initialize function is carefully designed to allow us to read through the
        training dataset only once, and not store it in memory. As such, it can't itself
        manually iterate over the data source. Instead, the initialize function is a
        coroutine, which is sent row data. This should look roughly like::

            # set up variables here
            ...
            try:
                # start reading through data source
                while True:
                    # row has type Dict[str, types.DataType]
                    row = yield
                    # update any variables, vocabularies, etc.
                    ...
            except GeneratorExit:
                # finalize your initialization, set instance variables, etc.
                ...

        See `WordTokenizer.initialize` for a more concrete example.
        """
        return
        # we need yield here to make this function a generator
        yield


class TokenTensorizer(Tensorizer):
    """Convert text to a list of tokens. Do this based on a tokenizer configuration,
    and build a vocabulary for numberization. Finally, pad the batch to create
    a square tensor of the correct size.
    """

    class Config(Tensorizer.Config):
        #: The name of the text column to parse from the data source.
        column: str = "text"
        #: The tokenizer to use to split input text into tokens.
        tokenizer: Tokenizer.Config = Tokenizer.Config()
        add_bos_token: bool = False
        add_eos_token: bool = False
        use_eos_token_for_bos: bool = False
        max_seq_len: Optional[int] = None

    @classmethod
    def from_config(cls, config: Config):
        tokenizer = create_component(ComponentType.TOKENIZER, config.tokenizer)
        return cls(
            text_column=config.column,
            tokenizer=tokenizer,
            add_bos_token=config.add_bos_token,
            add_eos_token=config.add_eos_token,
            use_eos_token_for_bos=config.use_eos_token_for_bos,
            max_seq_len=config.max_seq_len,
        )

    def __init__(
        self,
        text_column,
        tokenizer=None,
        add_bos_token=Config.add_bos_token,
        add_eos_token=Config.add_eos_token,
        use_eos_token_for_bos=Config.use_eos_token_for_bos,
        max_seq_len=Config.max_seq_len,
        vocab=None,
    ):
        super().__init__([(text_column, str)])
        self.text_column = text_column
        self.tokenizer = tokenizer or Tokenizer()
        self.vocab = vocab
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.use_eos_token_for_bos = use_eos_token_for_bos
        self.max_seq_len = max_seq_len or 2 ** 30  # large number
        self.vocab_builder = None

    def _lookup_tokens(self, text):
        tokenized = self.tokenizer.tokenize(text)[: self.max_seq_len]
        if self.add_bos_token:
            bos = EOS if self.use_eos_token_for_bos else BOS
            tokenized = [Token(bos, -1, -1)] + tokenized
        if self.add_eos_token:
            tokenized.append(Token(EOS, -1, -1))
        if not tokenized:
            tokenized = [Token(PAD, -1, -1)]

        tokenized_texts, start_idx, end_idx = zip(
            *((t.value, t.start, t.end) for t in tokenized)
        )
        tokens = self.vocab.lookup_all(tokenized_texts)
        return tokens, start_idx, end_idx

    def _reverse_lookup(self, token_ids):
        return [self.vocab[id] for id in token_ids]

    def initialize(self, vocab_builder=None):
        """Build vocabulary based on training corpus."""
        if self.vocab:
            return
        self.vocab_builder = vocab_builder or VocabBuilder()
        try:
            while True:
                row = yield
                raw_text = row[self.text_column]
                tokenized = self.tokenizer.tokenize(raw_text)
                self.vocab_builder.add_all([t.value for t in tokenized])
        except GeneratorExit:
            self.vocab = self.vocab_builder.make_vocab()

    def numberize(self, row):
        """Tokenize, look up in vocabulary."""
        tokens, start_idx, end_idx = self._lookup_tokens(row[self.text_column])
        token_ranges = list(zip(start_idx, end_idx))
        return tokens, len(tokens), token_ranges

    def tensorize(self, batch):
        tokens, seq_lens, token_ranges = zip(*batch)
        return (
            pad_and_tensorize(tokens, self.vocab.idx[PAD]),
            pad_and_tensorize(seq_lens),
            pad_and_tensorize(token_ranges),
        )

    def sort_key(self, row):
        # use seq_len as sort key
        return row[1]


class ByteTensorizer(Tensorizer):
    """Turn characters into sequence of int8 bytes. One character will have one
    or more bytes depending on it's encoding
    """

    UNK_BYTE = 0
    PAD_BYTE = 0
    NUM = 256

    class Config(Tensorizer.Config):
        #: The name of the text column to parse from the data source.
        column: str = "text"
        lower: bool = True
        max_seq_len: Optional[int] = None

    @classmethod
    def from_config(cls, config: Config):
        return cls(config.column, config.lower, config.max_seq_len)

    def __init__(self, text_column, lower=True, max_seq_len=None):
        super().__init__([(text_column, str)])
        self.text_column = text_column
        self.lower = lower
        self.max_seq_len = max_seq_len

    def numberize(self, row):
        """Convert text to characters."""
        text = row[self.text_column]
        if self.lower:
            text = text.lower()
        bytes = list(text.encode())
        if self.max_seq_len:
            bytes = bytes[: self.max_seq_len]
        return bytes, len(bytes)

    def tensorize(self, batch):
        bytes, bytes_len = zip(*batch)
        return pad_and_tensorize(bytes, self.PAD_BYTE), pad_and_tensorize(bytes_len)

    def sort_key(self, row):
        # use bytes_len as sort key
        return row[1]


class CharacterTokenTensorizer(TokenTensorizer):
    """Turn words into 2-dimensional tensors of ints based on their ascii values.
    Words are padded to the maximum word length (also capped at `max_char_length`).
    Sequence lengths are the length of each token, 0 for pad token.
    """

    class Config(TokenTensorizer.Config):
        #: The max character length for a token.
        max_char_length: int = 20

    def __init__(self, max_char_length: int = Config.max_char_length, **kwargs):
        self.max_char_length = max_char_length
        super().__init__(**kwargs)

    # Don't need to create a vocab
    initialize = Tensorizer.initialize

    def numberize(self, row):
        """Convert text to characters, pad batch."""
        tokens = self.tokenizer.tokenize(row[self.text_column])[: self.max_seq_len]
        characters = [
            self._numberize_token(token)[: self.max_char_length] for token in tokens
        ]
        token_lengths = len(tokens)
        char_lengths = [len(token_chars) for token_chars in characters]
        return characters, token_lengths, char_lengths

    def _numberize_token(self, token):
        return [ord(c) for c in token.value]

    def tensorize(self, batch):
        characters, token_lengths, char_lengths = zip(*batch)
        return (
            pad_and_tensorize(characters),
            pad_and_tensorize(token_lengths),
            pad_and_tensorize(char_lengths),
        )

    def sort_key(self, row):
        return len(row[0])


class LabelTensorizer(Tensorizer):
    """Numberize labels. Label can be used as either input or target """

    __EXPANSIBLE__ = True

    class Config(Tensorizer.Config):
        #: The name of the label column to parse from the data source.
        column: str = "label"
        #: Whether to allow for unknown labels at test/prediction time
        allow_unknown: bool = False
        #: if vocab should have pad, usually false when label is used as target
        pad_in_vocab: bool = False

    @classmethod
    def from_config(cls, config: Config):
        return cls(config.column, config.allow_unknown, config.pad_in_vocab)

    def __init__(
        self,
        label_column: str = "label",
        allow_unknown: bool = False,
        pad_in_vocab: bool = False,
    ):
        super().__init__([(label_column, str)])
        self.label_column = label_column
        self.allow_unknown = allow_unknown
        self.pad_in_vocab = pad_in_vocab

    def initialize(self):
        """
        Look through the dataset for all labels and create a vocab map for them.
        """
        builder = VocabBuilder()
        builder.use_pad = self.pad_in_vocab
        builder.use_unk = self.allow_unknown
        try:
            while True:
                row = yield
                labels = row[self.label_column]
                builder.add_all(labels)
        except GeneratorExit:
            self.vocab = builder.make_vocab()
            self.pad_idx = (
                self.vocab.idx[PAD]
                if self.pad_in_vocab
                else Padding.DEFAULT_LABEL_PAD_IDX
            )

    def numberize(self, row):
        """Numberize labels."""
        return self.vocab.lookup_all(row[self.label_column])

    def tensorize(self, batch):
        return pad_and_tensorize(batch, self.pad_idx)


class LabelListTensorizer(LabelTensorizer):
    """LabelListTensorizer takes a list of labels as input and generate a tuple
    of tensors (label_idx, list_length).
    """

    def __init__(self, label_column: str = "label", *args, **kwargs):
        super().__init__(label_column, *args, **kwargs)
        self.column_schema = [(label_column, List[str])]

    def numberize(self, row):
        labels = super().numberize(row)
        return labels, len(labels)

    def tensorize(self, batch):
        labels, labels_len = zip(*batch)
        return super().tensorize(labels), pad_and_tensorize(labels_len)

    def sort_key(self, row):
        # use list length as sort key
        return row[1]


class SoftLabelTensorizer(LabelTensorizer):
    """
    Handles numberizing labels for knowledge distillation. This still requires the same
    label column as `LabelTensorizer` for the "true" label, but also processes soft
    "probabilistic" labels generated from a teacher model, via three new columns.
    """

    class Config(LabelTensorizer.Config):
        probs_column: str = "target_probs"
        logits_column: str = "target_logits"
        labels_column: str = "target_labels"

    @classmethod
    def from_config(cls, config: Config):
        return cls(
            config.column,
            config.allow_unknown,
            config.pad_in_vocab,
            config.probs_column,
            config.logits_column,
            config.labels_column,
        )

    def __init__(
        self,
        label_column: str = "label",
        allow_unknown: bool = False,
        pad_in_vocab: bool = False,
        probs_column: str = "target_probs",
        logits_column: str = "target_logits",
        labels_column: str = "target_labels",
    ):
        column_schema = [
            (label_column, str),
            (probs_column, List[float]),
            (logits_column, List[float]),
            (labels_column, List[str]),
        ]
        Tensorizer.__init__(self, column_schema)
        self.label_column = label_column
        self.allow_unknown = allow_unknown
        self.pad_in_vocab = pad_in_vocab
        self.probs_column = probs_column
        self.logits_column = logits_column
        self.labels_column = labels_column

    def numberize(self, row):
        """Numberize hard and soft labels"""
        label = self.vocab.lookup_all(row[self.label_column])
        row_labels = row[self.labels_column]
        probs = align_target_label(row[self.probs_column], row_labels, self.vocab.idx)
        logits = align_target_label(row[self.logits_column], row_labels, self.vocab.idx)
        return label, probs, logits

    def tensorize(self, batch):
        label, probs, logits = zip(*batch)
        return (
            pad_and_tensorize(label, self.pad_idx),
            pad_and_tensorize(probs, dtype=torch.float),
            pad_and_tensorize(logits, dtype=torch.float),
        )


class NumericLabelTensorizer(Tensorizer):
    """Numberize numeric labels."""

    class Config(Tensorizer.Config):
        #: The name of the label column to parse from the data source.
        column: str = "label"
        #: If provided, the range of values the raw label can be. Will rescale the
        #: label values to be within [0, 1].
        rescale_range: Optional[List[float]] = None

    @classmethod
    def from_config(cls, config: Config):
        return cls(config.column, config.rescale_range)

    def __init__(
        self,
        label_column: str = Config.column,
        rescale_range: Optional[List[float]] = Config.rescale_range,
    ):
        super().__init__([(label_column, str)])
        self.label_column = label_column
        if rescale_range is not None:
            assert len(rescale_range) == 2
            assert rescale_range[0] < rescale_range[1]
        self.rescale_range = rescale_range

    def numberize(self, row):
        """Numberize labels."""
        label = float(row[self.label_column])
        if self.rescale_range is not None:
            label -= self.rescale_range[0]
            label /= self.rescale_range[1] - self.rescale_range[0]
            assert 0 <= label <= 1
        return label

    def tensorize(self, batch):
        return pad_and_tensorize(batch, dtype=torch.float)


class FloatListTensorizer(Tensorizer):
    """Numberize numeric labels."""

    class Config(Tensorizer.Config):
        #: The name of the label column to parse from the data source.
        column: str
        error_check: bool = False
        dim: Optional[int] = None

    @classmethod
    def from_config(cls, config: Config):
        return cls(config.column, config.error_check, config.dim)

    def __init__(self, column: str, error_check: bool, dim: Optional[int]):
        super().__init__([(column, str)])
        self.column = column
        self.error_check = error_check
        self.dim = dim
        assert not self.error_check or self.dim is not None, "Error check requires dim"

    def numberize(self, row):
        str = row[self.column]
        # replace spaces between float numbers with commas (regex101.com/r/C2705x/1)
        str = re.sub(r"(?<=[\d.])\s*,?\s+(?=[+-]?[\d.])", ",", str)
        # remove dot not followed with a digit (regex101.com/r/goSmuG/1/)
        str = re.sub(r"(?<=\d)\.(?![\d])", "", str)
        try:
            res = json.loads(str)
        except json.decoder.JSONDecodeError as e:
            raise Exception(
                f"Unable to parse dense feature:{row[self.column]}," f" re output:{str}"
            ) from e
        if type(res) is not list:
            raise ValueError(f"{res} is not a valid float list")
        if self.error_check:
            assert len(res) == self.dim, (
                f"Expected dimension:{self.dim}"
                f", got:{len(res)}"
                f", dense-feature:{res}"
            )
        return [float(n) for n in res]

    def tensorize(self, batch):
        return pad_and_tensorize(batch, dtype=torch.float)


NO_LABEL = SpecialToken("NO_LABEL")


class WordLabelTensorizer(Tensorizer):
    """Numberize word/slot labels."""

    class Config(Tensorizer.Config):
        #: The name of the slot label column to parse from the data source.
        slot_column: str = "slots"
        #: The name of the text column to parse from the data source.
        #: We need this to be able to generate tensors which correspond to input text.
        text_column: str = "text"
        #: The tokenizer to use to split input text into tokens. This should be
        #: configured in a way which yields tokens consistent with the tokens input to
        #: or output by a model, so that the labels generated by this tensorizer
        #: will match the indices of the model's tokens.
        tokenizer: Tokenizer.Config = Tokenizer.Config()
        #: Whether to allow for unknown labels at test/prediction time
        allow_unknown: bool = False

    @classmethod
    def from_config(cls, config: Component.Config):
        tokenizer = create_component(ComponentType.TOKENIZER, config.tokenizer)
        return cls(
            config.slot_column, config.text_column, tokenizer, config.allow_unknown
        )

    def __init__(
        self,
        slot_column: str = Config.slot_column,
        text_column: str = Config.text_column,
        tokenizer: Tokenizer = None,
        allow_unknown: bool = Config.allow_unknown,
    ):
        super().__init__([(text_column, str), (slot_column, List[Slot])])
        self.slot_column = slot_column
        self.text_column = text_column
        self.allow_unknown = allow_unknown
        self.tokenizer = tokenizer or Tokenizer()

    def initialize(self):
        """Look through the dataset for all labels and create a vocab map for them."""
        builder = VocabBuilder()
        builder.add(NO_LABEL)
        builder.use_unk = self.allow_unknown
        try:
            while True:
                row = yield
                slots = row[self.slot_column]
                builder.add_all(s.label for s in slots)
        except GeneratorExit:
            self.vocab = builder.make_vocab()

    def numberize(self, row):
        """
        Turn slot labels and text into a list of token labels with the same
        length as the number of tokens in the text.
        """
        slots = row[self.slot_column]
        text = row[self.text_column]
        tokens = self.tokenizer.tokenize(text)
        indexed_tokens = tokens
        labels = []
        current_slot = 0
        current_token = 0
        while current_token < len(tokens) and current_slot < len(slots):
            _, start, end = indexed_tokens[current_token]
            slot = slots[current_slot]
            if start > slot.end:
                current_slot += 1
            else:
                current_token += 1
                labels.append(slot.label if end > slot.start else NO_LABEL)
        return self.vocab.lookup_all(labels)

    def tensorize(self, batch):
        return pad_and_tensorize(batch, dtype=torch.long)


class GazetteerTensorizer(Tensorizer):
    """
    Create 3 tensors for dict features.

    - idx: index of feature in token order.
    - weights: weight of feature in token order.
    - lens: number of features per token.

    For each input token, there will be the same number of `idx` and `weights` entries.
    (equal to the max number of features any token has in this row). The values
    in `lens` will tell how many of these features are actually used per token.

    Input format for the dict column is json and should be a list of dictionaries
    containing the "features" and their weight for each relevant "tokenIdx". Example:
    ::

        text: "Order coffee from Starbucks please"
        dict: [
            {"tokenIdx": 1, "features": {"drink/beverage": 0.8, "music/song": 0.2}},
            {"tokenIdx": 3, "features": {"store/coffee_shop": 1.0}}
        ]

    if we assume this vocab
    ::

        vocab = {
            UNK: 0, PAD: 1,
            "drink/beverage": 2, "music/song": 3, "store/coffee_shop": 4
        }

    this example will result in those tensors:
    ::

        idx =     [1,   1,   2,   3,   1,   1,   4,   1,   1,   1]
        weights = [0.0, 0.0, 0.8, 0.2, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        lens =    [1,        2,        1,        1,        1]

    """

    class Config(Tensorizer.Config):
        text_column: str = "text"
        dict_column: str = "dict"
        #: tokenizer to split text and create dict tensors of the same size.
        tokenizer: Tokenizer.Config = Tokenizer.Config()

    @classmethod
    def from_config(cls, config: Config):
        return cls(config.column)

    def __init__(
        self,
        text_column: str = Config.text_column,
        dict_column: str = Config.dict_column,
        tokenizer: Tokenizer = None,
    ):
        super().__init__([(text_column, str), (dict_column, Gazetteer)])
        self.text_column = text_column
        self.dict_column = dict_column
        self.tokenizer = tokenizer or Tokenizer()

    def initialize(self):
        """
        Look through the dataset for all dict features to create vocab.
        """
        builder = VocabBuilder()
        try:
            while True:
                row = yield
                for token_dict in row[self.dict_column]:
                    builder.add_all(token_dict["features"])
        except GeneratorExit:
            self.vocab = builder.make_vocab()

    def numberize(self, row):
        """
        Numberize dict features. Fill in for tokens with no features with
        PAD and weight 0.0. All tokens need to have at least one entry.
        Tokens with more than one feature will have multiple idx and weight
        added in sequence.
        """

        num_tokens = len(self.tokenizer.tokenize(row[self.text_column]))
        num_labels = max(len(t["features"]) for t in row[self.dict_column])
        res_idx = [self.vocab.idx[PAD]] * (num_labels * num_tokens)
        res_weights = [0.0] * (num_labels * num_tokens)
        res_lens = [1] * num_tokens
        for dict_feature in row[self.dict_column]:
            idx = dict_feature["tokenIdx"]
            feats = dict_feature["features"]
            pos = idx * num_labels
            res_lens[idx] = len(feats)
            # write values at the correct pos
            for label, weight in feats.items():
                res_idx[pos] = self.vocab.lookup_all(label)
                res_weights[pos] = weight
                pos += 1

        return res_idx, res_weights, res_lens

    def tensorize(self, batch):
        dict_feat_ids, weights, seq_lens = zip(*batch)
        return (
            pad_and_tensorize(dict_feat_ids, self.vocab.idx[PAD]),
            pad_and_tensorize(weights, self.vocab.idx[PAD]),
            pad_and_tensorize(seq_lens),
        )


class RawString(Tensorizer):
    """A pass-through tensorizer to include raw fields from datasource in the batch.
       Used mostly for metric reporting."""

    class Config(Tensorizer.Config):
        #: The name of the pass-through column to parse from the data source.
        column: str

    @classmethod
    def from_config(cls, config: Config):
        return cls(config.column)

    def __init__(self, column: str):
        super().__init__([(column, str)])
        self.column = column

    def numberize(self, row):
        return row[self.column]


class JoinStringTensorizer(Tensorizer):
    """A pass-through tensorizer to include raw fields from datasource in the batch.
       Used mostly for metric reporting."""

    class Config(Tensorizer.Config):
        #: The name of the pass-through column to parse from the data source.
        columns: List[str]
        delimiter: str = " | "

    @classmethod
    def from_config(cls, config: Config):
        return cls(config.columns, config.delimiter)

    def __init__(self, columns: List[str], delimiter: str):
        super().__init__([(column, str) for column in columns])
        self.columns = columns
        self.delimiter = delimiter

    def numberize(self, row):
        return self.delimiter.join([row[column] for column in self.columns])


class RawJson(RawString):
    def numberize(self, row):
        return json.loads(row[self.column])


class MetricTensorizer(Tensorizer):
    """A tensorizer which use other tensorizers' numerized data.
       Used mostly for metric reporting."""

    class Config(Tensorizer.Config):
        names: List[str]
        indexes: List[int]

    @classmethod
    def from_config(cls, config: Config):
        return cls(config.names, config.indexes)

    def __init__(self, names: List[str], indexes: List[int]):
        super().__init__([])
        self.names = names
        self.indexes = indexes

    def numberize(self, row):
        # metric tensorizer will depends on other tensorizers' numeric result
        return None

    def tensorize(self, batch):
        raise NotImplementedError


class NtokensTensorizer(MetricTensorizer):
    """A tensorizer which will reference another tensorizer's numerized data
       to calculate the num tokens.
       Used for calculating tokens per second."""

    def tensorize(self, batch):
        ntokens = 0
        for name, index in zip(self.names, self.indexes):
            ntokens += sum((sample[index] for sample in batch[name]))
        return ntokens


def initialize_tensorizers(tensorizers, data_source):
    """A utility function to stream a data source to the initialize functions
    of a dict of tensorizers."""
    initializers = []
    for init in [tensorizer.initialize() for tensorizer in tensorizers.values()]:
        try:
            init.send(None)  # kick
            initializers.append(init)
        except StopIteration:
            pass

    if initializers:
        for row in data_source:
            for init in initializers:
                init.send(row)

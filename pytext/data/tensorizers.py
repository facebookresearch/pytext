#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import json
import re
from typing import List, Optional, Tuple, Type

import torch
from pytext.config.component import Component, ComponentType, create_component
from pytext.utils.data import Slot

from .utils import (
    BOS,
    EOS,
    PAD,
    SpecialToken,
    Token,
    Tokenizer,
    VocabBuilder,
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
        while True:
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

    def _lookup_tokens(self, text):
        tokenized = self.tokenizer.tokenize(text)[: self.max_seq_len]
        if self.add_bos_token:
            bos = EOS if self.use_eos_token_for_bos else BOS
            tokenized = [Token(bos, -1, -1)] + tokenized
        if self.add_eos_token:
            tokenized.append(Token(EOS, -1, -1))
        tokenized_texts, start_idx, end_idx = zip(
            *((t.value, t.start, t.end) for t in tokenized)
        )
        tokens = self.vocab.lookup_all(tokenized_texts)
        return tokens, start_idx, end_idx

    def _reverse_lookup(self, token_ids):
        return [self.vocab[id] for id in token_ids]

    def initialize(self):
        """Build vocabulary based on training corpus."""
        builder = VocabBuilder()
        try:
            while True:
                if self.vocab:
                    yield
                else:
                    row = yield
                    raw_text = row[self.text_column]
                    tokenized = self.tokenizer.tokenize(raw_text)
                    builder.add_all([t.value for t in tokenized])
        except GeneratorExit:
            if not self.vocab:
                self.vocab = builder.make_vocab()

    def numberize(self, row):
        """Tokenize, look up in vocabulary."""
        tokens, _, _ = self._lookup_tokens(row[self.text_column])
        return tokens, len(tokens)

    def tensorize(self, batch):
        tokens, seq_lens = zip(*batch)
        return (
            pad_and_tensorize(tokens, self.vocab.idx[PAD]),
            pad_and_tensorize(seq_lens),
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
    Words are padded to the maximum word length. Sequence lengths are the lengths
    of each token, 0 for pad token.
    """

    def numberize(self, row):
        """Convert text to characters, pad batch."""
        tokens = self.tokenizer.tokenize(row[self.text_column])
        lengths = [len(token.value) for token in tokens]
        characters = [[ord(c) for c in token.value] for token in tokens]
        return characters, lengths

    def tensorize(self, batch):
        characters, lengths = zip(*batch)
        return (pad_and_tensorize(characters), pad_and_tensorize(lengths))


class LabelTensorizer(Tensorizer):
    """Numberize labels."""

    class Config(Tensorizer.Config):
        #: The name of the label column to parse from the data source.
        column: str = "label"
        #: Whether to allow for unknown labels at test/prediction time
        allow_unknown: bool = False

    @classmethod
    def from_config(cls, config: Config):
        return cls(config.column, config.allow_unknown)

    def __init__(
        self, label_column: str = "label", allow_unknown: bool = Config.allow_unknown
    ):
        super().__init__([(label_column, str)])
        self.label_column = label_column
        self.allow_unknown = allow_unknown

    def initialize(self):
        """
        Look through the dataset for all labels and create a vocab map for them.
        """
        builder = VocabBuilder()
        builder.use_pad = False
        builder.use_unk = self.allow_unknown
        try:
            while True:
                row = yield
                labels = row[self.label_column]
                builder.add_all(labels.split(","))
        except GeneratorExit:
            self.labels = builder.make_vocab()

    def numberize(self, row):
        """Numberize labels."""
        return self.labels.lookup_all(row[self.label_column])

    def tensorize(self, batch):
        return pad_and_tensorize(batch)


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
        dim: Optional[int] = None

    @classmethod
    def from_config(cls, config: Config):
        return cls(config.column)

    def __init__(self, column: str):
        super().__init__([(column, str)])
        self.column = column

    def numberize(self, row):
        res = json.loads(re.sub(r",? +", ",", row[self.column]))
        if type(res) is not list:
            raise ValueError(f"{res} is not a valid float list")
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


class RawJson(RawString):
    def numberize(self, row):
        return json.loads(row[self.column])


def initialize_tensorizers(tensorizers, data_source):
    """A utility function to stream a data source to the initialize functions
    of a dict of tensorizers."""
    initializers = [tensorizer.initialize() for tensorizer in tensorizers.values()]
    for init in initializers:
        init.send(None)  # kick
    for row in data_source:
        for init in initializers:
            init.send(row)

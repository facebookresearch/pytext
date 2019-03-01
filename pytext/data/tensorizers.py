#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
from pytext.config.component import Component, ComponentType, create_component

from .utils import PAD, Tokenizer, VocabBuilder, pad


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

    @classmethod
    def from_config(cls, config: Config):
        return cls(config.column)

    def __init__(self, column):
        self.column = column

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


class WordTensorizer(Tensorizer):
    """Convert text to a list of tokens. Do this based on a tokenizer configuration,
    and build a vocabulary for numberization. Finally, pad the batch to create
    a square tensor of the correct size.
    """

    class Config(Tensorizer.Config):
        #: The name of the text column to parse from the data source.
        column: str = "text"
        #: The tokenizer to use to split input text into tokens.
        tokenizer: Tokenizer.Config = Tokenizer.Config()

    @classmethod
    def from_config(cls, config: Config):
        tokenizer = create_component(ComponentType.TOKENIZER, config.tokenizer)
        return cls(config.column, tokenizer)

    def __init__(self, column, tokenizer=None):
        super().__init__(column)
        self.tokenizer = tokenizer or Tokenizer()

    def initialize(self):
        """Build vocabulary based on training corpus."""
        builder = VocabBuilder()
        try:
            while True:
                row = yield
                raw_text = row[self.column]
                tokenized = self.tokenizer.tokenize(raw_text)
                builder.add_all([t.value for t in tokenized])
        except GeneratorExit:
            self.vocab = builder.make_vocab()

    def create_training_tensors(self, batch):
        """Tokenize, look up in vocabulary, and pad."""
        tokenized_texts = [
            [t.value for t in self.tokenizer.tokenize(row[self.column])]
            for row in batch
        ]
        tokens = self.vocab.lookup_all(tokenized_texts)
        seq_lens = [len(tokenized) for tokenized in tokenized_texts]
        padded_tokens = pad(tokens, self.vocab.idx[PAD])
        return (
            torch.tensor(padded_tokens, dtype=torch.long),
            torch.tensor(seq_lens, dtype=torch.long),
        )


class ByteTensorizer(Tensorizer):
    """Turn characters into ints based on their ascii values."""

    PAD_IDX = 0

    class Config(Tensorizer.Config):
        #: The name of the text column to parse from the data source.
        column: str = "text"

    def create_training_tensors(self, batch):
        """Convert text to characters, pad batch."""
        texts = [[ord(c) for c in row[self.column]] for row in batch]
        seq_lens = [len(text) for text in texts]
        padded_texts = pad(texts, self.PAD_IDX)
        return torch.LongTensor(padded_texts), torch.LongTensor(seq_lens)


class WordCharacterTensorizer(WordTensorizer):
    """Turn words into 2-dimensional tensors of ints based on their ascii values.
    Words are padded to the maximum word length. Sequence lengths are the lengths
    of each token, 0 for pad token.
    """

    PAD_IDX = 0

    def create_training_tensors(self, batch):
        """Convert text to characters, pad batch."""
        all_tokens = [self.tokenizer.tokenize(row[self.column]) for row in batch]
        lengths = [[len(token.value) for token in tokens] for tokens in all_tokens]
        characters = [
            [[ord(c) for c in token.value] for token in tokens] for tokens in all_tokens
        ]
        return (
            torch.LongTensor(pad(characters, self.PAD_IDX)),
            torch.LongTensor(pad(lengths, self.PAD_IDX)),
        )


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
        self, column: str = Config.column, allow_unknown: bool = Config.allow_unknown
    ):
        self.column = column
        self.allow_unknown = allow_unknown

    def initialize(self):
        """Look through the dataset for all labels and create a vocab map for them."""
        builder = VocabBuilder()
        builder.use_pad = False
        builder.use_unk = self.allow_unknown
        try:
            while True:
                row = yield
                labels = row[self.column]
                builder.add_all(labels.split(","))
        except GeneratorExit:
            self.labels = builder.make_vocab()

    def create_training_tensors(self, batch):
        """Numberize labels."""
        labels = [row[self.column] for row in batch]
        return torch.tensor(self.labels.lookup_all(labels), dtype=torch.long)


class MetaInput(Tensorizer):
    """A pass-through tensorizer to include raw fields from datasource in the batch.
       Used mostly for metric reporting."""

    class Config(Tensorizer.Config):
        #: The name of the text column to parse from the data source.
        column: str = "text"

    def create_training_tensors(self, batch):
        return [row[self.column] for row in batch]


def initialize_tensorizers(tensorizers, data_source):
    """A utility function to stream a data source to the initialize functions
    of a dict of tensorizers."""
    initializers = [tensorizer.initialize() for tensorizer in tensorizers.values()]
    for init in initializers:
        init.send(None)  # kick
    for row in data_source:
        for init in initializers:
            init.send(row)

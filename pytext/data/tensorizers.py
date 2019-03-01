#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from pytext.config.component import Component, ComponentType, create_component

from .utils import PAD, Tokenizer, VocabBuilder, pad_and_tensorize


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

    def numberize(self, row):
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

    def numberize(self, row):
        """Tokenize, look up in vocabulary."""
        tokenized_texts = [t.value for t in self.tokenizer.tokenize(row[self.column])]
        tokens = self.vocab.lookup_all(tokenized_texts)
        seq_len = len(tokenized_texts)
        return tokens, seq_len

    def tensorize(self, batch):
        tokens, seq_lens = zip(*batch)
        return (
            pad_and_tensorize(tokens, self.vocab.idx[PAD]),
            pad_and_tensorize(seq_lens),
        )


class ByteTensorizer(Tensorizer):
    """Turn characters into ints based on their ascii values."""

    class Config(Tensorizer.Config):
        #: The name of the text column to parse from the data source.
        column: str = "text"

    def numberize(self, row):
        """Convert text to characters."""
        text = [ord(c) for c in row[self.column]]
        seq_len = len(text)
        return text, seq_len

    def tensorize(self, batch):
        tokens, seq_lens = zip(*batch)
        return pad_and_tensorize(tokens, 0), pad_and_tensorize(seq_lens)


class WordCharacterTensorizer(WordTensorizer):
    """Turn words into 2-dimensional tensors of ints based on their ascii values.
    Words are padded to the maximum word length. Sequence lengths are the lengths
    of each token, 0 for pad token.
    """

    def initialize(self):
        while True:
            yield

    def numberize(self, row):
        """Convert text to characters, pad batch."""
        tokens = self.tokenizer.tokenize(row[self.column])
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

    def numberize(self, row):
        """Numberize labels."""
        return self.labels.lookup_all(row[self.column])

    def tensorize(self, batch):
        return pad_and_tensorize(batch)


class MetaInput(Tensorizer):
    """A pass-through tensorizer to include raw fields from datasource in the batch.
       Used mostly for metric reporting."""

    class Config(Tensorizer.Config):
        #: The name of the text column to parse from the data source.
        column: str = "text"

    def numberize(self, row):
        return row[self.column]


def initialize_tensorizers(tensorizers, data_source):
    """A utility function to stream a data source to the initialize functions
    of a dict of tensorizers."""
    initializers = [tensorizer.initialize() for tensorizer in tensorizers.values()]
    for init in initializers:
        init.send(None)  # kick
    for row in data_source:
        for init in initializers:
            init.send(row)

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from pytext.data.tensorizers import Tensorizer
from pytext.data.utils import VocabBuilder, pad_and_tensorize


class MyWordTensorizer(Tensorizer):
    """
    Simple Tensorizer that splits a sentence on spaces and create tensors
    from the vocabulary index built on the training data.
    """

    class Config(Tensorizer.Config):
        #: The name of the text column to parse from the data source.
        column: str = "text"

    @classmethod
    def from_config(cls, config: Config):
        return cls(column=config.column)

    def __init__(self, column):
        self.column = column
        self.vocab = None

    @property
    def column_schema(self):
        return [(self.column, str)]

    def _tokenize(self, row):
        raw_text = row[self.column]
        return raw_text.split()

    def initialize(self):
        """Build vocabulary based on training corpus."""
        vocab_builder = VocabBuilder()

        try:
            while True:
                row = yield
                words = self._tokenize(row)
                vocab_builder.add_all(words)
        except GeneratorExit:
            self.vocab = vocab_builder.make_vocab()

    def numberize(self, row):
        """Look up tokens in vocabulary to get their corresponding index"""
        words = self._tokenize(row)
        idx = self.vocab.lookup_all(words)
        # LSTM representations need the length of the sequence
        return idx, len(idx)

    def tensorize(self, batch):
        tokens, seq_lens = zip(*batch)
        return (
            pad_and_tensorize(tokens, self.vocab.get_pad_index()),
            pad_and_tensorize(seq_lens),
        )

    def sort_key(self, row):
        # LSTM representations need the batches to be sorted by descending seq_len
        return row[1]

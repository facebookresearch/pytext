#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
from random import Random
from typing import Dict, List, Type

from pytext.common.constants import SpecialTokens
from pytext.data.sources.data_source import RootDataSource
from pytext.utils.file_io import PathManager


def load_vocab(file_path):
    """
    Given a file, prepare the vocab dictionary where each line is the value and
    (line_no - 1) is the key
    """
    vocab = {}
    with PathManager.open(file_path, "r") as file_contents:
        for idx, word in enumerate(file_contents):
            vocab[str(idx)] = word.strip()
    return vocab


def reader(file_path, vocab):
    with PathManager.open(file_path, "r") as r:
        for line in r:
            yield " ".join(
                vocab.get(s.strip(), SpecialTokens.UNK)
                # In ATIS every query starts/ends with BOS/EOS: we remove them
                for s in line.split()[1:-1]
            )


def reader_raw(file_path, vocab):
    with PathManager.open(file_path, "r") as r:
        for line in r:
            yield vocab[line.strip()]


class AtisIntentDataSource(RootDataSource):
    """
    DataSource which loads queries and intent from the ATIS dataset.

    The simple usage is to provide the path the unzipped atis directory, and
    it will use the default filenames and parameters.

    ATIS dataset has the following characteristics:
    - words are stored in a dict file.
    - content files contain only indices of words.
    - there's no eval set: we'll take random rows from the train set.
    - all queries start with BOS (Beginning Of Sentence) and end with EOS
      (End Of Sentence), which we'll remove.
    """

    class Config(RootDataSource.Config):
        path: str = "."
        field_names: List[str] = ["text", "label"]
        validation_split: float = 0.25
        random_seed: int = 12345
        # Filenames can be overridden if necessary
        intent_filename: str = "atis.dict.intent.csv"
        vocab_filename: str = "atis.dict.vocab.csv"
        test_queries_filename: str = "atis.test.query.csv"
        test_intent_filename: str = "atis.test.intent.csv"
        train_queries_filename: str = "atis.train.query.csv"
        train_intent_filename: str = "atis.train.intent.csv"

    # Config mimics the constructor
    # This will be the default in future pytext.
    @classmethod
    def from_config(cls, config: Config, schema: Dict[str, Type]):
        return cls(schema=schema, **config._asdict())

    def __init__(
        self,
        path=Config.path,
        field_names=None,
        validation_split=Config.validation_split,
        random_seed=Config.random_seed,
        intent_filename=Config.intent_filename,
        vocab_filename=Config.vocab_filename,
        test_queries_filename=Config.test_queries_filename,
        test_intent_filename=Config.test_intent_filename,
        train_queries_filename=Config.train_queries_filename,
        train_intent_filename=Config.train_intent_filename,
        **kwargs,
    ):
        super().__init__(**kwargs)

        field_names = field_names or AtisIntentDataSource.Config.field_names
        assert (
            len(field_names or []) == 2
        ), "AtisIntentDataSource only handles 2 field_names: {}".format(field_names)

        self.query_field = field_names[0]
        self.intent_field = field_names[1]

        self.random_seed = random_seed
        self.validation_split = validation_split

        # Load the vocab dict in memory and provide a lookup function
        # This allows other applications to
        self.words = load_vocab(os.path.join(path, vocab_filename))
        self.intents = load_vocab(os.path.join(path, intent_filename))

        self.test_queries_filepath = os.path.join(path, test_queries_filename)
        self.test_intent_filepath = os.path.join(path, test_intent_filename)
        self.train_queries_filepath = os.path.join(path, train_queries_filename)
        self.train_intent_filepath = os.path.join(path, train_intent_filename)

    def _selector(self, select_eval):
        """
        This selector ensures that the same pseudo-random sequence is
        always the same from the beginning. The `select_eval` parameter
        guarantees that the training set and eval set are exact complements.
        """
        rng = Random(self.random_seed)

        def fn():
            return select_eval ^ (rng.random() >= self.validation_split)

        return fn

    def _iter_rows(self, query_reader, intent_reader, select_fn=lambda: True):
        for query_str, intent_str in zip(query_reader, intent_reader):
            if select_fn():
                yield {self.query_field: query_str, self.intent_field: intent_str}

    def raw_train_data_generator(self):
        return iter(
            self._iter_rows(
                query_reader=reader(self.train_queries_filepath, self.words),
                intent_reader=reader_raw(self.train_intent_filepath, self.intents),
                select_fn=self._selector(select_eval=False),
            )
        )

    def raw_eval_data_generator(self):
        return iter(
            self._iter_rows(
                query_reader=reader(self.train_queries_filepath, self.words),
                intent_reader=reader_raw(self.train_intent_filepath, self.intents),
                select_fn=self._selector(select_eval=True),
            )
        )

    def raw_test_data_generator(self):
        return iter(
            self._iter_rows(
                query_reader=reader(self.test_queries_filepath, self.words),
                intent_reader=reader_raw(self.test_intent_filepath, self.intents),
            )
        )


if __name__ == "__main__":
    import sys

    src = AtisIntentDataSource(sys.argv[1], field_names=["query", "intent"], schema={})
    for row in src.raw_train_data_generator():
        print("TRAIN", row)
    for row in src.raw_eval_data_generator():
        print("EVAL", row)
    for row in src.raw_test_data_generator():
        print("TEST", row)

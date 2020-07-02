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
    with PathManager.open(file_path, "r") as reader:
        for line in reader:
            yield " ".join(
                vocab.get(s.strip(), SpecialTokens.UNK)
                # ATIS every row starts/ends with BOS/EOS: remove them
                for s in line.split()[1:-1]
            )


class AtisSlotsDataSource(RootDataSource):
    """
    DataSource which loads queries and slots from the ATIS dataset.

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
        field_names: List[str] = ["text", "slots"]
        validation_split: float = 0.25
        random_seed: int = 12345
        # Filenames can be overridden if necessary
        slots_filename: str = "atis.dict.slots.csv"
        vocab_filename: str = "atis.dict.vocab.csv"
        test_queries_filename: str = "atis.test.query.csv"
        test_slots_filename: str = "atis.test.slots.csv"
        train_queries_filename: str = "atis.train.query.csv"
        train_slots_filename: str = "atis.train.slots.csv"

    # Config mimics the constructor

    @classmethod
    def from_config(cls, config: Config, schema: Dict[str, Type]):
        return cls(schema=schema, **config._asdict())

    def __init__(
        self,
        path=Config.path,
        field_names=None,
        validation_split=Config.validation_split,
        random_seed=Config.random_seed,
        slots_filename=Config.slots_filename,
        vocab_filename=Config.vocab_filename,
        test_queries_filename=Config.test_queries_filename,
        test_slots_filename=Config.test_slots_filename,
        train_queries_filename=Config.train_queries_filename,
        train_slots_filename=Config.train_slots_filename,
        **kwargs,
    ):
        super().__init__(**kwargs)

        field_names = field_names or ["text", "slots"]
        assert (
            len(field_names or []) == 2
        ), "AtisSlotsDataSource only handles 2 field_names: {}".format(field_names)

        self.random_seed = random_seed
        self.validation_split = validation_split

        # Load the vocab dict in memory and provide a lookup function
        self.words = load_vocab(os.path.join(path, vocab_filename))
        self.slots = load_vocab(os.path.join(path, slots_filename))

        self.query_field = field_names[0]
        self.slots_field = field_names[1]

        self.test_queries_filepath = os.path.join(path, test_queries_filename)
        self.test_slots_filepath = os.path.join(path, test_slots_filename)
        self.train_queries_filepath = os.path.join(path, train_queries_filename)
        self.train_slots_filepath = os.path.join(path, train_slots_filename)

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

    def _iter_rows(self, query_reader, slots_reader, select_fn=lambda: True):
        for query_str, slots_str in zip(query_reader, slots_reader):
            if select_fn():
                yield {self.query_field: query_str, self.slots_field: slots_str}

    def raw_train_data_generator(self):
        return iter(
            self._iter_rows(
                query_reader=reader(self.train_queries_filepath, self.words),
                slots_reader=reader(self.train_slots_filepath, self.slots),
                select_fn=self._selector(select_eval=False),
            )
        )

    def raw_eval_data_generator(self):
        return iter(
            self._iter_rows(
                query_reader=reader(self.train_queries_filepath, self.words),
                slots_reader=reader(self.train_slots_filepath, self.slots),
                select_fn=self._selector(select_eval=True),
            )
        )

    def raw_test_data_generator(self):
        return iter(
            self._iter_rows(
                query_reader=reader(self.test_queries_filepath, self.words),
                slots_reader=reader(self.test_slots_filepath, self.slots),
            )
        )


if __name__ == "__main__":
    import sys

    src = AtisSlotsDataSource(sys.argv[1], field_names=["text", "slots"], schema={})
    for row in src.raw_train_data_generator():
        print("TRAIN", row)
    for row in src.raw_eval_data_generator():
        print("EVAL", row)
    for row in src.raw_test_data_generator():
        print("TEST", row)

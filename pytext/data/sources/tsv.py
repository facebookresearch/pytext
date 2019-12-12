#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import csv
import sys
import threading
from typing import Dict, List, Optional, Type

from pytext.utils.path import get_absolute_path

from .data_source import (
    RootDataSource,
    SafeFileWrapper,
    ShardedDataSource,
    generator_property,
)
from .session import SessionDataSource


class TSV:
    def __init__(
        self,
        file,
        field_names=None,
        delimiter="\t",
        quoted=False,
        drop_incomplete_rows=False,
    ):
        self.file = file
        self.field_names = field_names
        self.delimiter = delimiter
        self.quoted = quoted
        self.drop_incomplete_rows = drop_incomplete_rows
        self.total_rows_count = 0
        self.incomplete_rows_count = 0
        self._access_lock = threading.Lock()
        csv.field_size_limit(sys.maxsize)

    def __iter__(self):
        can_acquire = self._access_lock.acquire(blocking=False)
        if not can_acquire:
            raise Exception("Concurrent iteration not supported")
        self.file.seek(0)
        try:
            reader = csv.DictReader(
                (line.replace("\0", "") for line in self.file),
                fieldnames=self.field_names,
                delimiter=self.delimiter,
                quoting=csv.QUOTE_MINIMAL if self.quoted else csv.QUOTE_NONE,
            )
            if self.drop_incomplete_rows:
                for row in reader:
                    self.total_rows_count += 1
                    if any(map(lambda v: v is None, row.values())):  # drop!
                        self.incomplete_rows_count += 1
                        continue
                    yield row
            else:
                yield from reader
        finally:
            self._access_lock.release()

    def __del__(self):
        print("Destroying TSV object")
        print(f"Total number of rows read: {self.total_rows_count}")
        print(f"Total number of rows dropped: {self.incomplete_rows_count}")


class TSVDataSource(RootDataSource):
    """DataSource which loads data from TSV sources. Uses python's csv library."""

    class Config(RootDataSource.Config):
        #: Filename of training set. If not set, iteration will be empty.
        train_filename: Optional[str] = None
        #: Filename of testing set. If not set, iteration will be empty.
        test_filename: Optional[str] = None
        #: Filename of eval set. If not set, iteration will be empty.
        eval_filename: Optional[str] = None
        #: Field names for the TSV. If this is not set, the first line of each file
        #: will be assumed to be a header containing the field names.
        field_names: Optional[List[str]] = None
        #: The column delimiter passed to Python's csv library. Change to "," for csv.
        delimiter: str = "\t"
        #: Whether the columns can use quotes to include delimiters or not.
        #: Rows with unclosed quotes will be merged with \n inside.
        #: Change to True for quoted csv.
        quoted: bool = False
        # Flag to turn on dropping rows with columns less than the expected
        # number of columns. This prevents passing None/null as column values
        # down to the tensorizer.
        drop_incomplete_rows: bool = False

    @classmethod
    def from_config(cls, config: Config, schema: Dict[str, Type], **kwargs):
        args = config._asdict()
        train_filename = args.pop("train_filename")
        test_filename = args.pop("test_filename")
        eval_filename = args.pop("eval_filename")

        train_file = (
            SafeFileWrapper(
                get_absolute_path(train_filename), encoding="utf-8", errors="replace"
            )
            if train_filename
            else None
        )
        test_file = (
            SafeFileWrapper(
                get_absolute_path(test_filename), encoding="utf-8", errors="replace"
            )
            if test_filename
            else None
        )
        eval_file = (
            SafeFileWrapper(
                get_absolute_path(eval_filename), encoding="utf-8", errors="replace"
            )
            if eval_filename
            else None
        )
        return cls(
            train_file=train_file,
            test_file=test_file,
            eval_file=eval_file,
            schema=schema,
            **args,
            **kwargs,
        )

    def __init__(
        self,
        train_file=None,
        test_file=None,
        eval_file=None,
        field_names=None,
        delimiter=Config.delimiter,
        quoted=Config.quoted,
        drop_incomplete_rows=Config.drop_incomplete_rows,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._init_tsv(
            field_names,
            delimiter,
            train_file,
            test_file,
            eval_file,
            quoted,
            drop_incomplete_rows,
        )

    def _init_tsv(
        self,
        field_names,
        delimiter,
        train_file,
        test_file,
        eval_file,
        quoted,
        drop_incomplete_rows,
    ):
        def make_tsv(file):
            return TSV(
                file,
                field_names=field_names,
                delimiter=delimiter,
                quoted=quoted,
                drop_incomplete_rows=drop_incomplete_rows,
            )

        self._train_tsv = make_tsv(train_file) if train_file else []
        self._test_tsv = make_tsv(test_file) if test_file else []
        self._eval_tsv = make_tsv(eval_file) if eval_file else []

    def raw_train_data_generator(self):
        return iter(self._train_tsv)

    def raw_test_data_generator(self):
        return iter(self._test_tsv)

    def raw_eval_data_generator(self):
        return iter(self._eval_tsv)


class MultilingualTSVDataSource(TSVDataSource):
    """
    Data Source for multi-lingual data. The input data can have multiple
    text fields and each field can either have the same language or different
    languages. The `data_source_languages` dict contains the language information
    for each text field and this should match the number of language identifiers
    specified in `language_columns`.
    """

    class Config(TSVDataSource.Config):
        data_source_languages: Dict[str, List[str]] = {
            "train": ["en"],
            "eval": ["en"],
            "test": ["en"],
        }
        language_columns: List[str] = ["language"]

    def __init__(
        self,
        train_file=None,
        test_file=None,
        eval_file=None,
        field_names=None,
        delimiter=Config.delimiter,
        data_source_languages=Config.data_source_languages,
        language_columns=Config.language_columns,
        **kwargs,
    ):
        super().__init__(
            train_file, test_file, eval_file, field_names, delimiter, **kwargs
        )
        self.data_source_languages = data_source_languages
        self.language_columns = language_columns
        assert len(data_source_languages["train"]) == len(
            self.language_columns
        ), "Number of languages and language columns should be the same."

    def _convert_raw_source(self, source, languages):
        for row in source:
            example = self._read_example(row)
            if example is None:
                continue
            for col, lang in zip(self.language_columns, languages):
                example[col] = lang
            yield example

    @generator_property
    def train(self):
        return self._convert_raw_source(
            self.raw_train_data_generator(), self.data_source_languages["train"]
        )

    @generator_property
    def test(self):
        return self._convert_raw_source(
            self.raw_test_data_generator(), self.data_source_languages["test"]
        )

    @generator_property
    def eval(self):
        return self._convert_raw_source(
            self.raw_eval_data_generator(), self.data_source_languages["eval"]
        )


class BlockShardedTSV:
    """Take a TSV file, split into N pieces (by byte location) and return
    an iterator on one of the pieces.  The pieces are equal by byte size,
    not by number of rows.  Thus, care needs to be taken when using this
    for distributed training, otherwise number of batches for different
    workers might be different.
    """

    def __init__(
        self,
        file,
        field_names=None,
        delimiter="\t",
        quoted=False,
        block_id=0,
        num_blocks=1,
        drop_incomplete_rows=False,
    ):
        self.file = file
        self.field_names = field_names
        self.delimiter = delimiter
        self.quoted = quoted
        self.block_id = block_id
        self.num_blocks = num_blocks
        self.drop_incomplete_rows = drop_incomplete_rows
        csv.field_size_limit(sys.maxsize)

    def __iter__(self):
        # (self.begin, self.end) are the pointers to the begin and end
        # of file segment
        self.file.seek(0, 2)
        end = self.file.tell()
        self.begin = self.block_id * end / self.num_blocks
        self.end = (self.block_id + 1) * end / self.num_blocks
        self.file.seek(self.begin, 0)
        # make sure we're at the beginning of a full row
        if self.begin:
            self.file.readline()
        reader = csv.DictReader(
            (line.replace("\0", "") for line in iter(self.file.readline, "")),
            fieldnames=self.field_names,
            delimiter=self.delimiter,
            quoting=csv.QUOTE_MINIMAL if self.quoted else csv.QUOTE_NONE,
        )
        # iterate until we're at the end of segment
        for row in reader:
            if self.file.tell() > self.end:
                break
            if self.drop_incomplete_rows:
                if any(map(lambda v: v is None, row.values())):  # drop!
                    continue
            yield row


class BlockShardedTSVDataSource(TSVDataSource, ShardedDataSource):
    def __init__(self, rank=0, world_size=1, **kwargs):
        self.rank = rank
        self.world_size = world_size
        # calls init of TSVDataSource
        super().__init__(**kwargs)
        # weird python syntax to call init of ShardedDataSource
        super(TSVDataSource, self).__init__(schema=self.schema)

    def _init_tsv(
        self,
        field_names,
        delimiter,
        train_file,
        test_file,
        eval_file,
        quoted,
        drop_incomplete_rows,
    ):
        def make_tsv(file, rank=0, world_size=1):
            return BlockShardedTSV(
                file,
                field_names=field_names,
                delimiter=delimiter,
                block_id=rank,
                num_blocks=world_size,
                quoted=quoted,
                drop_incomplete_rows=drop_incomplete_rows,
            )

        self._train_tsv = (
            make_tsv(train_file, self.rank, self.world_size) if train_file else []
        )
        self._test_tsv = make_tsv(test_file) if test_file else []
        self._eval_tsv = make_tsv(eval_file) if eval_file else []
        self._train_unsharded = (
            TSV(train_file, field_names=field_names, delimiter=delimiter, quoted=quoted)
            if train_file
            else []
        )

    @generator_property
    def train_unsharded(self):
        return iter(self._train_unsharded)


class SessionTSVDataSource(TSVDataSource, SessionDataSource):
    def __init__(
        self,
        train_file=None,
        test_file=None,
        eval_file=None,
        field_names=None,
        **kwargs,
    ):
        # requires first column to be the session id
        assert len(field_names) >= 2, "should specify at least 2 columns"
        super().__init__(
            train_file=train_file,
            test_file=test_file,
            eval_file=eval_file,
            field_names=field_names,
            id_col=field_names[0],
            **kwargs,
        )
        self.schema[self.id_col] = str
        self.field_names = field_names
        self._validate_schema()

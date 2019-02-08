#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import csv
import threading
from typing import List, Optional

from pytext.data import types

from .data_source import DataSchema, RootDataSource, SafeFileWrapper


class TSV:
    def __init__(self, file, field_names=None, delimiter="\t"):
        self.file = file
        self.field_names = field_names
        self.delimiter = delimiter
        self._access_lock = threading.Lock()

    def __iter__(self):
        can_acquire = self._access_lock.acquire(blocking=False)
        if not can_acquire:
            raise Exception("Concurrent iteration not supported")
        self.file.seek(0)
        try:
            reader = csv.DictReader(
                self.file, fieldnames=self.field_names, delimiter=self.delimiter
            )
            yield from reader
        finally:
            self._access_lock.release()


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

    @classmethod
    def from_config(cls, config: Config, schema: DataSchema):
        args = config._asdict()
        train_filename = args.pop("train_filename")
        test_filename = args.pop("test_filename")
        eval_filename = args.pop("eval_filename")
        train_file = SafeFileWrapper(train_filename) if train_filename else None
        test_file = SafeFileWrapper(test_filename) if test_filename else None
        eval_file = SafeFileWrapper(eval_filename) if eval_filename else None
        return cls(train_file, test_file, eval_file, schema=schema, **args)

    def __init__(
        self,
        train_file=None,
        test_file=None,
        eval_file=None,
        field_names=None,
        delimiter=Config.delimiter,
        **kwargs,
    ):
        super().__init__(**kwargs)

        def make_tsv(file):
            return TSV(file, field_names=field_names, delimiter=delimiter)

        self._train_tsv = make_tsv(train_file) if train_file else []
        self._test_tsv = make_tsv(test_file) if test_file else []
        self._eval_tsv = make_tsv(eval_file) if eval_file else []
        self._access_lock = threading.Lock()

    def _iter_raw_train(self):
        return iter(self._train_tsv)

    def _iter_raw_test(self):
        return iter(self._test_tsv)

    def _iter_raw_eval(self):
        return iter(self._eval_tsv)


@TSVDataSource.register_type(types.Text)
def load_text(s):
    return types.Text(s)


@TSVDataSource.register_type(types.Label)
def load_label(s):
    return types.Label(s)

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from itertools import chain
from typing import List

from pytext.config.serialize import _get_class_type
from pytext.data.sources.data_source import RootDataSource


class SessionDataSource(RootDataSource):
    """
    Data source for session based data, the input data is organized in sessions,
    each session may have multiple rows. The first column is always the session id.
    Raw input rows are consolidated by session id and returned as one session
    per example
    """

    def __init__(self, id_col, **kwargs):
        self.id_col = id_col
        self.current_id = None
        self.current_session = []
        super().__init__(**kwargs)

    def _validate_schema(self):
        """Make sure the input schema are all list type, which is the return value
        type, and convert it to the actual type (e.g List[T] -> T) when reading the
        raw data from file.
        """
        for k, v in self.schema.items():
            if k != self.id_col:
                assert _get_class_type(v) is list, f"{k} is not a list type!"
                self.schema[k] = v.__args__[0]

    def merge_session(self, session):
        res = {self.id_col: session[0][self.id_col]}
        for k, v in chain.from_iterable([s.items() for s in session]):
            if k != self.id_col:
                res[k] = res.get(k, [])
                res[k].append(v)
        return res

    def _convert_raw_source(self, source):
        for row in source:
            example = self._read_example(row)
            if example is None:
                continue
            if example[self.id_col] == self.current_id:
                self.current_session.append(example)
            else:
                self.current_id = example[self.id_col]
                session = self.current_session
                self.current_session = [example]
                if session:
                    yield self.merge_session(session)
        self.current_id = None
        session = self.current_session
        self.current_session = []
        if session:
            yield self.merge_session(session)

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
from typing import Dict

from pytext.config.component import Component, ComponentType
from pytext.data import types


class RawExample(dict):
    """A wrapper class for a single example row with a dict interface.
    This is here for any logic we want row objects to have that dicts don't do.
    The values of this dictionary are `pytext.data.types.DataType` objects."""


# Map of registered types for data source subclasses
DATA_SOURCE_TYPES = {}


DataSchema = Dict[str, types.DataType]
DataSchemaConfig = Dict[str, types.DataType.Config]


class SafeFileWrapper:
    """A simple wrapper class for files which allows filedescriptors to be managed
    with normal Python ref counts. Python doesn't do this anymore manually because
    the python standard doesn't guarantee garbage collection semantics, but this
    will work properly in 99% of cases. If you are seeing resource warnings on your
    system, please file a github issue."""

    def __init__(self, *args, **kwargs):
        self._file = open(*args, **kwargs)

    def __del__(self):
        self._file.close()

    def __iter__(self):
        """Some file utilities check hasattr(o, "__iter__") explicitly."""
        return iter(self._file)

    def __getattr__(self, attr):
        return getattr(self._file, attr)


class GeneratorIterator:
    """Create an object which can be iterated over multiple times from a
    generator call. Each iteration will call the generator and allow iterating
    over it. This is unsafe to use on generators which have side effects, such
    as file readers; it's up to the callers to safely manage these scenarios.
    """

    def __init__(self, generator, *args, **kwargs):
        self.generator = generator
        self.args = args
        self.kwargs = kwargs

    def __iter__(self):
        return self.generator(*self.args, **self.kwargs)


class GeneratorMethodProperty:
    """Identify a generator method as a property. This will allow instances to iterate
    over the property multiple times, and not consume the generator. It accomplishes
    this by wrapping the generator and creating multiple generator instances if
    iterated over multiple times.
    """

    def __init__(self, generator):
        self.generator = generator

    def __get__(self, obj, objtype=None):
        return GeneratorIterator(self.generator, obj)


# Use the more typical property decorator style
generator_property = GeneratorMethodProperty


class DataSource(Component):
    """
    Data sources are simple components that stream data from somewhere using Python's
    iteration interface. It should expose 3 iterators, "train", "test", and "eval".
    Each of these should be able to be iterated over any number of times, and iterating
    over it should yield dictionaries whose values are instances of
    `pytext.data.types.DataType`.

    Simply, these data sources exist as an interface to read through datasets
    in a pythonic way, with pythonic types, and abstract away the form that they are
    stored in.
    """

    __COMPONENT_TYPE__ = ComponentType.DATA_SOURCE
    __EXPANSIBLE__ = True

    def __init__(self, schema):
        self.schema = schema

    @generator_property
    def train(self):
        raise NotImplementedError

    @generator_property
    def test(self):
        raise NotImplementedError

    @generator_property
    def eval(self):
        raise NotImplementedError


class RootDataSource(DataSource):
    """A data source which actually loads data from a location. This data source
    needs to be responsible for converting types based on a schema, because it should
    be the only part of the system that actually needs to understand details about
    the underlying storage system.

    A RootDataSource should implement `_iter_raw_train`, `_iter_raw_test`,
    and `_iter_raw_eval`. These functions should yield dictionaries of raw objects
    which the loading system can convert using the schema loading functions. If this
    is not a helpful abstraction, feel free to just subclass DataSource and implement
    `train`, `test`, and `eval` directly.
    """

    class Config(Component.Config):
        #: An optional column mapping, allowing the columns in the raw data source
        #: to not map directly to the column names in the schema. This mapping will
        #: remap names from the raw data source to names in the schema.
        column_mapping: Dict[str, str] = {}

    def __init__(self, schema, column_mapping=()):
        super().__init__(schema)
        self.column_mapping = dict(column_mapping)

    def _convert_raw_source(self, source):
        """Convert a raw iterable source, ie. from `DataSource._iter_raw_train`,
        to an iterable that will yield `pytext.data.type.DataType` objects
        according to the schema and the converters for this DataSource.
        """
        for row in source:
            example = RawExample()
            for column_name, value in row.items():
                name = self.column_mapping.get(column_name, column_name)
                if name not in self.schema:
                    continue
                example[name] = self.load(value, self.schema[name])
            if len(example) != len(self.schema):
                # We might need to re-evaluate this for multi-task training
                logging.warn("Skipping row missing values")
                continue
            yield example

    @classmethod
    def register_type(cls, type):
        def decorator(fn):
            DATA_SOURCE_TYPES[(cls, type)] = fn
            return fn

        return decorator

    def load(self, value, schema_type):
        # It would be nice for subclasses of data sources to work better with this
        converter = DATA_SOURCE_TYPES[(type(self), schema_type)]
        return converter(value)

    def _iter_raw_train(self):
        raise NotImplementedError

    def _iter_raw_test(self):
        raise NotImplementedError

    def _iter_raw_eval(self):
        raise NotImplementedError

    @generator_property
    def train(self):
        return self._convert_raw_source(self._iter_raw_train())

    @generator_property
    def test(self):
        return self._convert_raw_source(self._iter_raw_test())

    @generator_property
    def eval(self):
        return self._convert_raw_source(self._iter_raw_eval())

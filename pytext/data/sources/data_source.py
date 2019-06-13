#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import json
import logging
from typing import Dict, List, TypeVar

from pytext.config.component import Component, ComponentType
from pytext.data.utils import shard
from pytext.utils.data import Slot, parse_slot_string


class RawExample(dict):
    """A wrapper class for a single example row with a dict interface.
    This is here for any logic we want row objects to have that dicts don't do."""


class SafeFileWrapper:
    """
    A simple wrapper class for files which allows filedescriptors to be managed
    with normal Python ref counts.
    Without using this, if you create a file in a from_config you will see a warning
    along the lines of "ResourceWarning: self._file is acquired but not always released"
    this is because we're opening a file not in a context manager (with statement).
    We want to do it this way because it lets us pass a file object to the DataSource,
    rather than a filename. This exposes a ton more flexibility and testability, passing
    filenames is one of the paths towards pain.

    However, we don't have a clear resource management system set up for configuration.
    from_config functions are the tool that we have to allow objects to specify how they
    should be created from a configuration, which generally should only happen from the
    command line, whereas in eg. a notebook you should build the objects with
    constructors directly. If building from constructors, you can just open a file and
    pass it, but from_config here needs to create a file object from a configured
    filename. Python files don't close automatically, so you also need a system that
    will close them when the python interpreter shuts down. If you don't, it will print
    a resource warning at runtime, as the interpreter manually closes the filehandles
    (although modern OSs are pretty okay with having open file handles, it's hard for me
    to justify exactly why Python is so strict about this; I think one of the main
    reasons you might actually care is if you have a writeable file handle it might not
    have flushed properly when the C runtime exits, but Python doesn't actually
    distinguish between writeable and non-writeable file handles).

    This class is a wrapper that creates a system for (sort-of) safely closing the file
    handles before the runtime exits. It does this by closing the file when the object's
    deleter is called. Although the python standard doesn't actually make any guarantees
    about when deleters are called, CPython is reference counted and so as an
    mplementation detail will call a deleter whenever the last reference to it is
    removed, which generally will happen to all objects created during program execution
    as long as there aren't reference cycles (I don't actually know off-hand whether the
    cycle collection is run before shutdown, and anyway the cycles would have to include
    objects that the runtime itself maintains pointers to, which seems like you'd have
    to work hard to do and wouldn't do accidentally). This isn't true for other python
    systems like PyPy or Jython which use generational garbage collection and so don't
    actually always call destructors before the system shuts down, but again this is
    only really relevant for mutable files.

    An alternative implementation would be to build a resource management system into
    PyText, something like a function that we use for opening system resources that
    registers the resources and then we make sure are all closed before system shutdown.
    That would probably technically be the right solution, but I didn't really think of
    that first and also it's a bit longer to implement.

    If you are seeing resource warnings on your system, please file a github issue.
    """

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
    over it should yield dictionaries whose values are deserialized python types.

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


class ShardedDataSource(DataSource):
    """Base class for sharded data sources."""


class RowShardedDataSource(ShardedDataSource):
    "Shards a given datasource by row."

    def __init__(self, data_source: DataSource, rank=0, world_size=1):
        super().__init__(data_source.schema)
        self.data_source = data_source
        self.rank = rank
        self.world_size = world_size
        self.eval = data_source.eval
        self.test = data_source.test

    @generator_property
    def train(self):
        return shard(iter(self.data_source.train), self.rank, self.world_size)

    @generator_property
    def train_unsharded(self):
        """Used to initialize tensorizer on the intire dataset."""
        return iter(self.data_source.train)


class RootDataSource(DataSource):
    """A data source which actually loads data from a location. This data source
    needs to be responsible for converting types based on a schema, because it should
    be the only part of the system that actually needs to understand details about
    the underlying storage system.

    RootDataSource presents a simpler abstraction than DataSource where the rows
    are automatically converted to the right DataTypes.

    A RootDataSource should implement `raw_train_data_generator`,
    `raw_test_data_generator`, and `raw_eval_data_generator`. These functions
    should yield dictionaries of raw objects which the loading system can
    convert using the schema loading functions.
    """

    DATA_SOURCE_TYPES = {}

    class Config(Component.Config):
        #: An optional column mapping, allowing the columns in the raw data source
        #: to not map directly to the column names in the schema. This mapping will
        #: remap names from the raw data source to names in the schema.
        column_mapping: Dict[str, str] = {}

    def __init__(self, schema, column_mapping=()):
        super().__init__(schema)
        self.column_mapping = dict(column_mapping)

    def _read_example(self, row):
        example = RawExample()
        for column_name, value in row.items():
            name = self.column_mapping.get(column_name, column_name)
            if name not in self.schema:
                continue
            example[name] = self.load(value, self.schema[name])
        if len(example) != len(self.schema):
            # We might need to re-evaluate this for multi-task training
            logging.warning(
                "Skipping row missing values: row {} -> schema {}".format(
                    list(row.keys()), list(self.schema.keys())
                )
            )
            return None
        return example

    def _convert_raw_source(self, source):
        """Convert a raw iterable source, ie. from
        `DataSource.raw_train_data_generator`, to an iterable that will yield
        `pytext.data.type.DataType` objects according to the schema and the
        converters for this DataSource.
        """
        for row in source:
            example = self._read_example(row)
            if example is None:
                continue
            yield example

    @classmethod
    def register_type(cls, type):
        def decorator(fn):
            cls.DATA_SOURCE_TYPES[type] = fn
            return fn

        return decorator

    def load(self, value, schema_type):
        if schema_type in self.DATA_SOURCE_TYPES:
            converter = self.DATA_SOURCE_TYPES[schema_type]
            return converter(value)
        try:
            return super().load(value, schema_type)
        except AttributeError:  # super lower than RootDataSource without load()
            raise Exception(
                'Type not registered in data source: "{}"'.format(schema_type)
            )

    def raw_train_data_generator(self):
        """
        Returns a generator that yields the TRAIN data one item at a time
        in a dictionary where each key is a field and the value is of the
        raw type from the source.
        DataSources need to implement this.
        """
        raise NotImplementedError

    def raw_test_data_generator(self):
        """
        Returns a generator that yields the TEST data one item at a time
        in a dictionary where each key is a field and the value is of the
        raw type from the source.
        DataSources need to implement this.
        """
        raise NotImplementedError

    def raw_eval_data_generator(self):
        """
        Returns a generator that yields the EVAL data one item at a time
        in a dictionary where each key is a field and the value is of the
        raw type from the source.
        DataSources need to implement this.
        """
        raise NotImplementedError

    @generator_property
    def train(self):
        return self._convert_raw_source(self.raw_train_data_generator())

    @generator_property
    def test(self):
        return self._convert_raw_source(self.raw_test_data_generator())

    @generator_property
    def eval(self):
        return self._convert_raw_source(self.raw_eval_data_generator())


@RootDataSource.register_type(str)
def load_text(s):
    return s


@RootDataSource.register_type(List[Slot])
def load_slots(s):
    return parse_slot_string(s)


Gazetteer = List[Dict[str, Dict[str, float]]]
JSONString = TypeVar("JSONString", str, bytes)


@RootDataSource.register_type(Gazetteer)
@RootDataSource.register_type(List[float])
@RootDataSource.register_type(List[str])
@RootDataSource.register_type(List[int])
def load_json(s):
    return json.loads(s)


@RootDataSource.register_type(JSONString)
def load_json_string(s):
    parsed = json.loads(s)
    if not isinstance(parsed, str):
        raise TypeError(
            "Expected input to be parsed into a string object. "
            + f"Got {type(parsed)} type.\n"
            + f"Original: <<{s}>>, Parsed: <<{parsed}>>"
        )
    return parsed


@RootDataSource.register_type(float)
def load_float(f):
    return float(f)

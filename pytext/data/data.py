#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import functools
import itertools
from typing import Dict, Iterable

from pytext.common.constants import Stage
from pytext.config.component import Component, ComponentType, create_component

from .sources import DataSchema, DataSource, RawExample
from .sources.data_source import GeneratorIterator
from .tensorizers import Tensorizer


class Batcher(Component):
    """Batcher designed to batch rows of data, before padding."""

    __COMPONENT_TYPE__ = ComponentType.BATCHER

    class Config(Component.Config):
        #: Make batches of this size when possible. If there's not enough data,
        #: might generate some smaller batches.
        batch_size: int = 16

    @classmethod
    def from_config(cls, config: Config):
        return cls(config.batch_size)

    def __init__(self, batch_size=Config.batch_size):
        self.batch_size = batch_size

    def batchify(self, iterable: Iterable[RawExample]):
        """Group rows by batch_size.  Assume iterable of dicts, yield dict of lists.
        The last batch will be of length len(iterable) % batch_size."""
        iterators = [iter(iterable)] * self.batch_size
        for batch in itertools.zip_longest(*iterators):
            yield zip_dicts([ex for ex in batch if ex is not None])


def numberize_rows(tensorizers, rows):
    for row in rows:
        yield {
            name: tensorizer.numberize(row) for name, tensorizer in tensorizers.items()
        }


def pad_and_tensorize_batches(tensorizers, batches):
    for batch in batches:
        yield {
            name: tensorizer.tensorize(batch[name])
            for name, tensorizer in tensorizers.items()
        }


def zip_dicts(dicts):
    all_keys = set(itertools.chain.from_iterable(dicts))
    zipped = {key: [] for key in all_keys}
    for d in dicts:
        for key in all_keys:
            zipped[key].append(d.get(key))
    return zipped


def generator_iterator(fn):
    """Turn a generator into a GeneratorIterator-wrapped function.
    Effectively this allows iterating over a generator multiple times by recording
    the call arguments, and calling the generator with them anew each item __iter__
    is called on the returned object."""

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        return GeneratorIterator(fn, *args, **kwargs)

    return wrapped


class Data(Component):
    """Data is an abstraction that handles all of the following:

    - Initialize model metadata parameters
    - Create batches of tensors for model training or prediction

    It can accomplish these in any way it needs to. The base implementation
    utilizes `pytext.data.sources.DataSource`, and sends batches to
    `pytext.data.tensorizers.Tensorizer` to create tensors.

    The `tensorizers` dict passed to the initializer should be considered something like
    a signature for the model. Each batch should be a dictionary with the same keys
    as the `tensorizers` dict, and values should be tensors arranged in the way
    specified by that tensorizer. The tensorizers dict doubles as a simple baseline
    implementation of that same signature, but subclasses of Data can override the
    implementation using other methods. This value is how the model specifies what
    inputs it's looking for.
    """

    __COMPONENT_TYPE__ = ComponentType.DATA_HANDLER
    __EXPANSIBLE__ = True

    class Config(Component.Config):
        #: Specify where training/test/eval data come from. The default value
        #: will not provide any data.
        source: DataSource.Config = DataSource.Config()
        #: How training examples are split into batches for the optimizer.
        batcher: Batcher.Config = Batcher.Config()

    @classmethod
    def from_config(
        cls, config: Config, schema: DataSchema, tensorizers: Dict[str, Tensorizer]
    ):
        data_source = create_component(ComponentType.DATA_SOURCE, config.source, schema)
        batcher = create_component(ComponentType.BATCHER, config.batcher)
        return cls(data_source, tensorizers, batcher=batcher)

    def __init__(
        self,
        data_source: DataSource,
        tensorizers: Dict[str, Tensorizer],
        batcher: Batcher = None,
    ):
        """This function should also initialize the passed in tensorizers with
        metadata they need for model construction."""
        self.data_source = data_source
        self.tensorizers = tensorizers
        self.batcher = batcher or Batcher()
        self.__initialize_tensorizers()

    def __initialize_tensorizers(self):
        """Initialize tensorizers using data from self.data_source.train."""
        initializers = [
            tensorizer.initialize() for tensorizer in self.tensorizers.values()
        ]
        for initializer in initializers:
            initializer.send(None)  # kick
        for row in self.data_source.train:
            for initializer in initializers:
                initializer.send(row)

    @generator_iterator
    def batches(self, stage: Stage):
        """Create batches of tensors to pass to model train_batch.
        This function yields dictionaries that mirror the `tensorizers` dict passed to
        `__init__`, ie. the keys will be the same, and the tensors will be the shape
        expected from the respective tensorizers.

        `stage` is used to determine which data source is used to create batches.
        """
        rows = {
            Stage.TRAIN: self.data_source.train,
            Stage.TEST: self.data_source.test,
            Stage.EVAL: self.data_source.eval,
        }[stage]

        numberized_rows = numberize_rows(self.tensorizers, rows)
        batches = self.batcher.batchify(numberized_rows)
        return pad_and_tensorize_batches(self.tensorizers, batches)

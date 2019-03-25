#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import functools
import itertools
import math
import random
from typing import Dict, Iterable, Optional, Type

from pytext.common.constants import Stage
from pytext.config.component import Component, ComponentType, create_component

from .sources import DataSource, RawExample, TSVDataSource
from .sources.data_source import GeneratorIterator
from .tensorizers import Tensorizer


class Batcher(Component):
    """Batcher designed to batch rows of data, before padding."""

    __COMPONENT_TYPE__ = ComponentType.BATCHER
    __EXPANSIBLE__ = True

    class Config(Component.Config):
        #: Make batches of this size when possible. If there's not enough data,
        #: might generate some smaller batches.
        train_batch_size: int = 16
        eval_batch_size: int = 16
        test_batch_size: int = 16
        #: Load this many batches of data together for shuffling.
        shuffle_num_batches: Optional[int] = 10000

    @classmethod
    def from_config(cls, config: Config):
        return cls(
            config.train_batch_size,
            config.eval_batch_size,
            config.test_batch_size,
            config.shuffle_num_batches,
        )

    def __init__(
        self,
        train_batch_size=Config.train_batch_size,
        eval_batch_size=Config.eval_batch_size,
        test_batch_size=Config.test_batch_size,
        shuffle_num_batches=Config.shuffle_num_batches,
    ):
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.test_batch_size = test_batch_size
        self._batch_sizes = {
            Stage.TRAIN: self.train_batch_size,
            Stage.TEST: self.eval_batch_size,
            Stage.EVAL: self.test_batch_size,
        }
        self.shuffle_num_batches = shuffle_num_batches or 1

    def batchify(
        self, iterable: Iterable[RawExample], sort_key=None, stage=Stage.TRAIN
    ):
        """
        Group rows by batch_size. Assume iterable of dicts, yield dict of lists.
        Will shuffle every group of `shuffle_num_batches` * `batch_size` rows, before
        batching. If sort key is provided, will sort rows within one batch.
        """
        batch_size = self._batch_sizes[stage]
        pool_size = self.shuffle_num_batches * batch_size
        for pool in self._group_iter(iterable, pool_size, shuffle=True):
            for batch in self._group_iter(pool, batch_size, sort_key=sort_key):
                yield zip_dicts(batch)

    def _group_iter(
        self, iterable: Iterable[RawExample], group_size, sort_key=None, shuffle=False
    ):
        iterators = [iter(iterable)] * group_size
        for group in itertools.zip_longest(*iterators):
            group = [ex for ex in group if ex is not None]
            if sort_key:
                group.sort(key=sort_key, reverse=True)
            elif shuffle:
                random.shuffle(group)
            yield group


class PoolingBatcher(Batcher):
    """
    Batcher that sorts and batches data rows, then shuffles the batches, before padding.
    This provides more efficient padding than the regular Batcher, when the given
    sort_key sorts by length.
    """

    def batchify(
        self, iterable: Iterable[RawExample], sort_key=None, stage=Stage.TRAIN
    ):
        """
        From an iterable of dicts, yield dicts of lists, by

        1. Load pool of shuffle_num_batches * batch_size examples.
        2. Sort rows, if necessary.
        3. Form batches with batch_size examples each.
        4. Shuffle batches and yield all batches.
        """
        batch_size = self._batch_sizes[stage]
        pool_size = self.shuffle_num_batches * batch_size

        for pool in self._group_iter(iterable, pool_size, sort_key=sort_key):
            batch_indices = list(range(math.ceil(len(pool) / batch_size)))
            if sort_key:
                random.shuffle(batch_indices)
            else:
                random.shuffle(pool)
            for batch_index in batch_indices:
                batch = pool[batch_size * batch_index : batch_size * (batch_index + 1)]
                yield zip_dicts(batch)


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


def shard(rows, rank, num_workers):
    """Only return every num_workers example for distributed training."""
    queue = []
    for row in rows:
        queue.append(row)
        # might discard remainder %num_workers rows because distributed
        # training needs to be in sync
        if len(queue) == num_workers:
            yield queue[rank]
            queue = []


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

    class Config(Component.Config):
        #: Specify where training/test/eval data come from. The default value
        #: will not provide any data.
        source: DataSource.Config = TSVDataSource.Config()
        #: How training examples are split into batches for the optimizer.
        batcher: Batcher.Config = PoolingBatcher.Config()
        sort_key: Optional[str] = None

    @classmethod
    def from_config(
        cls, config: Config, schema: Dict[str, Type], tensorizers: Dict[str, Tensorizer]
    ):
        data_source = create_component(ComponentType.DATA_SOURCE, config.source, schema)
        batcher = create_component(ComponentType.BATCHER, config.batcher)
        return cls(data_source, tensorizers, batcher=batcher, sort_key=config.sort_key)

    def __init__(
        self,
        data_source: DataSource,
        tensorizers: Dict[str, Tensorizer],
        batcher: Batcher = None,
        sort_key: Optional[str] = None,
    ):
        """This function should also initialize the passed in tensorizers with
        metadata they need for model construction."""
        self.data_source = data_source
        self.tensorizers = tensorizers
        self.batcher = batcher or Batcher()
        self.sort_key = sort_key
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
    def batches(self, stage: Stage, rank=0, world_size=1):
        """Create batches of tensors to pass to model train_batch.
        This function yields dictionaries that mirror the `tensorizers` dict passed to
        `__init__`, ie. the keys will be the same, and the tensors will be the shape
        expected from the respective tensorizers.

        `stage` is used to determine which data source is used to create batches.
        """
        rows = shard(
            {
                Stage.TRAIN: self.data_source.train,
                Stage.TEST: self.data_source.test,
                Stage.EVAL: self.data_source.eval,
            }[stage],
            rank,
            world_size,
        )

        numberized_rows = numberize_rows(self.tensorizers, rows)
        batches = self.batcher.batchify(
            numberized_rows,
            sort_key=(
                lambda row: self.tensorizers[self.sort_key].sort_key(row[self.sort_key])
            )
            if self.sort_key
            else None,
        )
        return pad_and_tensorize_batches(self.tensorizers, batches)

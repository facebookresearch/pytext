#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import functools
import itertools
import math
import multiprocessing
import queue
import random
from ctypes import c_bool
from typing import Any, Dict, Iterable, List, MutableMapping, NamedTuple, Optional, Type

from pytext.common.constants import Stage
from pytext.config.component import Component, ComponentType, Registry, create_component

from .sources import DataSource, RawExample, TSVDataSource
from .sources.data_source import (
    GeneratorIterator,
    RowShardedDataSource,
    ShardedDataSource,
)
from .tensorizers import MetricTensorizer, Tensorizer, initialize_tensorizers


# Number of batches to prefetch when loading batch data asynchronously
PREFETCH_QUEUE_SIZE: int = 10
# Number of seconds between waking to check if the prefetch generator exited
PREFETCH_QUEUE_WAKE_PERIOD: int = 1


class RowData(NamedTuple):
    raw_data: RawExample
    numberized: RawExample


class BatchData(NamedTuple):
    raw_data: List[RawExample]
    numberized: Dict[str, List[Any]]


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

    @classmethod
    def from_config(cls, config: Config):
        return cls(
            config.train_batch_size, config.eval_batch_size, config.test_batch_size
        )

    def __init__(
        self,
        train_batch_size=Config.train_batch_size,
        eval_batch_size=Config.eval_batch_size,
        test_batch_size=Config.test_batch_size,
    ):
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.test_batch_size = test_batch_size
        self._batch_sizes = {
            Stage.TRAIN: self.train_batch_size,
            Stage.TEST: self.test_batch_size,
            Stage.EVAL: self.eval_batch_size,
        }

    def batchify(
        self, iterable: Iterable[RawExample], sort_key=None, stage=Stage.TRAIN
    ):
        """Group rows by batch_size.  Assume iterable of dicts, yield dict of lists.
        The last batch will be of length len(iterable) % batch_size."""
        batch_size = self._batch_sizes[stage]
        for batch in self._group_iter(iterable, batch_size, sort_key):
            raw_batch, numberized_batch = zip(*batch)
            yield BatchData(raw_batch, zip_dicts(numberized_batch))

    def _group_iter(self, iterable: Iterable[RawExample], group_size, sort_key=None):
        iterators = [iter(iterable)] * group_size
        for group in itertools.zip_longest(*iterators):
            group = [ex for ex in group if ex is not None]
            if sort_key:
                group.sort(key=sort_key, reverse=True)
            yield group


class PoolingBatcher(Batcher):
    """
    Batcher that looks at pools of data, and sorts, batches, and shuffles them, before
    padding.
    """

    class Config(Batcher.Config):
        #: Number of batches in a pool, to load at one time.
        pool_num_batches: int = 10000

    @classmethod
    def from_config(cls, config: Config):
        return cls(
            config.train_batch_size,
            config.eval_batch_size,
            config.test_batch_size,
            config.pool_num_batches,
        )

    def __init__(
        self,
        train_batch_size=Config.train_batch_size,
        eval_batch_size=Config.eval_batch_size,
        test_batch_size=Config.test_batch_size,
        pool_num_batches=Config.pool_num_batches,
    ):
        super().__init__(train_batch_size, eval_batch_size, test_batch_size)
        self.pool_num_batches = pool_num_batches or 1

    def batchify(
        self, iterable: Iterable[RawExample], sort_key=None, stage=Stage.TRAIN
    ):
        """
        From an iterable of dicts, yield dicts of lists, by

        1. Load pool of batch_size * pool_num_batches examples.
        2. Sort rows, if necessary.
        3. Form batches with batch_size examples each.
        4. Shuffle batches and yield all batches.
        """
        batch_size = self._batch_sizes[stage]
        pool_size = batch_size * self.pool_num_batches

        for pool in self._group_iter(iterable, pool_size, sort_key):
            batch_indices = list(range(math.ceil(len(pool) / batch_size)))
            if sort_key:
                random.shuffle(batch_indices)
            else:
                random.shuffle(pool)
            for batch_index in batch_indices:
                batch = pool[batch_size * batch_index : batch_size * (batch_index + 1)]
                raw_batch, numberized_batch = zip(*batch)
                yield BatchData(raw_batch, zip_dicts(numberized_batch))


def prefetch(iterable, queue_size: int = PREFETCH_QUEUE_SIZE):
    """A prefetch mechanism for iterables which executes the iterator asynchronously
    in a separate process and waits on them. This can apply generally to any iterable.

    Args:
        iterable: Any iterable. The output of this function will have the same
                  iteration contract as iter(iterable).
        queue_size: The maximum number of values of the iterable to prefetch. This
                    allows a computation buffer while simultaneously ensuring we won't
                    completely expend the underlying iterable and load it all.
    Yields:
        The same values in the same order as the underlying generator.
    """
    cache = multiprocessing.Queue(maxsize=queue_size)
    iterator_empty = multiprocessing.Value(c_bool, False)

    def prefetch_thread(iterable, queue, iterator_empty):
        iterator = iter(iterable)
        while True:
            try:
                value = next(iterator)
            except StopIteration:
                iterator_empty.value = True
                return
            cache.put(value)

    process = multiprocessing.Process(
        target=prefetch_thread, args=(iterable, cache, iterator_empty)
    )
    process.start()

    try:
        while not iterator_empty.value:
            # don't wait forever, if we're pulling from queue faster than the worker
            # thread can generate data, it won't set iterator_empty in time before we
            # start waiting on get, and we'll wait forever.
            try:
                yield cache.get(timeout=PREFETCH_QUEUE_WAKE_PERIOD)
            except queue.Empty:
                pass
    except BaseException:  # If anything bad happened, clean up the child process
        if process.is_alive():
            process.terminate()
        raise
    else:
        process.join()


def pad_and_tensorize_batches(tensorizers, batches):
    for raw_batch, numberized_batch in batches:
        tensor_dict = {}
        for name, tensorizer in tensorizers.items():
            if isinstance(tensorizer, MetricTensorizer):
                tensor_dict[name] = tensorizer.tensorize(numberized_batch)
            else:
                tensor_dict[name] = tensorizer.tensorize(numberized_batch[name])

        yield raw_batch, tensor_dict


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
        source: DataSource.Config = TSVDataSource.Config()
        #: How training examples are split into batches for the optimizer.
        batcher: Batcher.Config = PoolingBatcher.Config()
        sort_key: Optional[str] = None
        #: cache numberized result in memory, turn off when CPU memory bound.
        in_memory: Optional[bool] = True

    @classmethod
    def from_config(
        cls,
        config: Config,
        schema: Dict[str, Type],
        tensorizers: Dict[str, Tensorizer],
        rank=0,
        world_size=1,
        **kwargs,
    ):
        data_source_cls = Registry.get(ComponentType.DATA_SOURCE, type(config.source))
        if issubclass(data_source_cls, ShardedDataSource):
            # data source is already sharded, we don't need to wrap RowShardedDataSource
            data_source = create_component(
                ComponentType.DATA_SOURCE,
                config.source,
                schema,
                rank=rank,
                world_size=world_size,
            )
        else:
            unsharded_data_source = create_component(
                ComponentType.DATA_SOURCE, config.source, schema
            )
            data_source = RowShardedDataSource(
                data_source=unsharded_data_source, rank=rank, world_size=world_size
            )

        batcher = create_component(ComponentType.BATCHER, config.batcher)
        return cls(
            data_source,
            tensorizers,
            batcher=batcher,
            sort_key=config.sort_key,
            in_memory=config.in_memory,
            **kwargs,
        )

    def __init__(
        self,
        data_source: DataSource,
        tensorizers: Dict[str, Tensorizer],
        batcher: Batcher = None,
        sort_key: Optional[str] = None,
        in_memory: Optional[bool] = False,
    ):
        """This function should also initialize the passed in tensorizers with
        metadata they need for model construction."""
        self.data_source = data_source
        self.tensorizers = tensorizers
        self.batcher = batcher or Batcher()
        self.sort_key = sort_key
        self.in_memory = in_memory
        self._numberized_cache: MutableMapping[str, Any] = {}
        full_train_data = (
            data_source.train_unsharded
            if isinstance(data_source, ShardedDataSource)
            else data_source.train
        )
        initialize_tensorizers(self.tensorizers, full_train_data)

    def _numberize_rows_no_cache(self, rows):
        """Numberize rows using tensorizers. Treat as a generator unless there's
        a good reason to believe the total dataset size is small. This will create
        objects for each input row in the dataset as it's iterated."""
        for row in rows:
            numberized = {
                name: tensorizer.numberize(row)
                for name, tensorizer in self.tensorizers.items()
            }
            yield RowData(row, numberized)

    def numberize_rows(self, rows, stage=None):
        """Get numberized rows, possibly from cache."""
        if self.in_memory:
            if stage not in self._numberized_cache:
                self._numberized_cache[stage] = list(
                    self._numberize_rows_no_cache(rows)
                )
            return self._numberized_cache[stage]
        return self._numberize_rows_no_cache(rows)

    @generator_iterator
    def batches(self, stage: Stage, data_source=None):
        """Create batches of tensors to pass to model train_batch.
        This function yields dictionaries that mirror the `tensorizers` dict passed to
        `__init__`, ie. the keys will be the same, and the tensors will be the shape
        expected from the respective tensorizers.

        `stage` is used to determine which data source is used to create batches.
        if data_source is provided, it is used instead of the configured data_sorce
        this is to allow setting a different data_source for testing a model.
        """
        data_source = data_source or self.data_source
        rows = {
            Stage.TRAIN: data_source.train,
            Stage.TEST: data_source.test,
            Stage.EVAL: data_source.eval,
        }[stage]

        # rows and numberized_rows are generators which can iterate over large
        # datasets; be careful not to do any operations which will expend them.
        numberized_rows = self.numberize_rows(rows, stage)
        sort_key = self.sort_key

        def key(row):
            return self.tensorizers[sort_key].sort_key(row.numberized[sort_key])

        batches = self.batcher.batchify(
            numberized_rows, sort_key=(key if sort_key else None)
        )
        batches = prefetch(batches)
        return pad_and_tensorize_batches(self.tensorizers, batches)

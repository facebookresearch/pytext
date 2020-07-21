#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import functools
import itertools
import math
import random
from typing import Any, Dict, Iterable, List, MutableMapping, NamedTuple, Optional, Type

from pytext.common.constants import RawExampleFieldName, Stage
from pytext.config.component import Component, ComponentType, Registry, create_component
from pytext.utils.usage import log_class_usage

from .sources import DataSource, RawExample, TSVDataSource
from .sources.data_source import (
    GeneratorIterator,
    RowShardedDataSource,
    ShardedDataSource,
)
from .tensorizers import MetricTensorizer, Tensorizer, initialize_tensorizers


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
    Batcher that shuffles and (if requested) sorts data.

    **Rationale**

    There is a trade-off between having batches of data that are truly randomly
    shuffled, and batches of data that are efficiently padded. If we wanted
    to maximise the efficiency of padding (i.e. minimise the amount of padding
    that is needed), we would have to enforce that all inputs of a similar
    length appear in the same batch. This however would lead to a dramatic
    decrease in the randomness of batches. On the other end of the spectrum,
    if we wanted to maximise randomness, we would often end up with inputs of
    wildly different lengths in the same batch, which would lead to a lot of
    padding.

    **Operation**

    This batcher uses a multi-staged approach.

    1. It first loads a number of "pools" of data, and shuffles them (this is
       controlled by `num_shuffled_pools`).
    2. It then splits up the shuffled data sequentially into individual pools,
       and the examples within each pool are sorted (if requested).
    3. Finally, each pool is split up sequentially into batches, and yielded.
       If sorting was requested in step #2, the order in which the batches are
       yielded is randomised.

    The size of a pool is expressed as a multiple of the batch size, and is
    controlled by `pool_num_batches`.


    **Examples**

    Assuming sorting is enabled, with the default settings of
    `pool_num_batches: 1000` and `num_shuffled_pools: 1`, a pool of
    `1k * batch_size` examples is loaded, sorted by length, and split up into
    1k batches. These batches are then yielded in random order. Once they run
    out, a new pool is loaded, and the process is repeated. An advantage of
    this approach is that padding will be somewhat reduced. A disadvantage is
    that, for every epoch, the first 1k batches will be always the same (albeit
    in a different order).

    On the other hand, specifying `pool_num_batches: 1000` and
    `num_shuffled_pools: 1000` would achieve the following:
    `1k * 1k * batch_size` examples are loaded, and shuffled. These are then
    split up into pools of size `1k * batch_size`, which are then sorted
    internally, split into individual batches, and yielded in random order.
    Compared to the previous example, we no longer have the problem that the
    first 1k batches are always the same in each epoch, but we've had to load
    in memory 1M examples.
    """

    class Config(Batcher.Config):
        #: Size of a pool expressed in number of batches
        pool_num_batches: int = 1000
        #: How many pool-sized chunks to load at a time for shuffling
        num_shuffled_pools: int = 1

    @classmethod
    def from_config(cls, config: Config):
        return cls(
            config.train_batch_size,
            config.eval_batch_size,
            config.test_batch_size,
            config.pool_num_batches,
            config.num_shuffled_pools,
        )

    def __init__(
        self,
        train_batch_size=Config.train_batch_size,
        eval_batch_size=Config.eval_batch_size,
        test_batch_size=Config.test_batch_size,
        pool_num_batches=Config.pool_num_batches,
        num_shuffled_pools=Config.num_shuffled_pools,
    ):
        super().__init__(train_batch_size, eval_batch_size, test_batch_size)
        assert pool_num_batches >= 1 and num_shuffled_pools >= 1
        self.pool_num_batches = pool_num_batches
        self.num_shuffled_pools = num_shuffled_pools

    def get_batch_size(self, stage: Stage) -> int:
        return self._batch_sizes[stage]

    def batchify(
        self, iterable: Iterable[RawExample], sort_key=None, stage=Stage.TRAIN
    ):
        """
        From an iterable of dicts, yield dicts of lists:

        1. Load `num_shuffled_pools` pools of data, and shuffle them.
        2. Load a pool (`batch_size * pool_num_batches` examples).
        3. Sort rows, if necessary.
        4. Shuffle the order in which the batches are returned, if necessary.
        """
        batch_size = self.get_batch_size(stage)
        pool_size = batch_size * self.pool_num_batches
        super_pool_size = pool_size * self.num_shuffled_pools

        for super_pool in self._group_iter(iterable, super_pool_size, None):
            # No point in shuffling if we're loading a single pool which is then sorted.
            if self.num_shuffled_pools > 1 or sort_key is None:
                random.shuffle(super_pool)
            for pool in self._group_iter(super_pool, pool_size, sort_key):
                batch_indices = list(range(math.ceil(len(pool) / batch_size)))
                if sort_key:
                    random.shuffle(batch_indices)
                for batch_index in batch_indices:
                    batch = pool[
                        batch_size * batch_index : batch_size * (batch_index + 1)
                    ]
                    raw_batch, numberized_batch = zip(*batch)
                    yield BatchData(raw_batch, zip_dicts(numberized_batch))


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
        init_tensorizers=True,
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
            init_tensorizers=init_tensorizers,
            **kwargs,
        )

    def __init__(
        self,
        data_source: DataSource,
        tensorizers: Dict[str, Tensorizer],
        batcher: Batcher = None,
        sort_key: Optional[str] = None,
        in_memory: Optional[bool] = True,
        init_tensorizers: Optional[bool] = True,
        init_tensorizers_from_scratch: Optional[bool] = True,
    ):
        """This function should also initialize the passed in tensorizers with
        metadata they need for model construction."""
        self.data_source = data_source
        self.tensorizers = tensorizers
        self.batcher = batcher or Batcher()
        self.sort_key = sort_key
        self.in_memory = in_memory
        self.numberized_cache: MutableMapping[str, Any] = {}
        self.cache_mutex: Dict[str, bool] = {}
        full_train_data = (
            data_source.train_unsharded
            if isinstance(data_source, ShardedDataSource)
            else data_source.train
        )
        if init_tensorizers:
            initialize_tensorizers(
                self.tensorizers, full_train_data, init_tensorizers_from_scratch
            )
        else:
            print(
                "Skipped initializing tensorizers since they are loaded from a "
                "previously saved state."
            )
        log_class_usage(__class__)

    def numberize_rows(self, rows):
        for row in rows:
            numberized = {
                name: tensorizer.numberize(row)
                for name, tensorizer in self.tensorizers.items()
            }
            yield RowData(row, numberized)

    def cache(self, numberized_rows, stage):
        if stage in self.cache_mutex:
            # already have generator caching the numberized data
            for numberized_row in numberized_rows:
                yield numberized_row
        else:
            self.cache_mutex[stage] = True
            result = []
            for numberized_row in numberized_rows:
                result.append(numberized_row)
                yield numberized_row
            self.numberized_cache[stage] = result

    def add_row_indices(self, rows):
        for idx, row in enumerate(rows):
            row[RawExampleFieldName.ROW_INDEX] = idx
            yield row

    @generator_iterator
    def batches(self, stage: Stage, data_source=None, load_early=False):
        """Create batches of tensors to pass to model train_batch.
        This function yields dictionaries that mirror the `tensorizers` dict passed to
        `__init__`, ie. the keys will be the same, and the tensors will be the shape
        expected from the respective tensorizers.

        `stage` is used to determine which data source is used to create batches.
        if data_source is provided, it is used instead of the configured data_sorce
        this is to allow setting a different data_source for testing a model.

        Passing in `load_early` = True disables loading all data in memory and using
        PoolingBatcher, so that we get the first batch as quickly as possible.
        """
        data_source = data_source if data_source is not None else self.data_source
        rows = {
            Stage.TRAIN: data_source.train,
            Stage.TEST: data_source.test,
            Stage.EVAL: data_source.eval,
        }[stage]

        # We add row indices here so that the original order can be reproduced
        # after shuffling the data if necessary.
        indexed_rows = self.add_row_indices(rows)

        # rows and numberized_rows are generators which can iterate over large
        # datasets; be careful not to do any operations which will expend them.
        if self.in_memory and not load_early:
            numberized_rows = self.numberized_cache.get(stage, None)
            if numberized_rows is None:
                numberized_rows = self.cache(self.numberize_rows(indexed_rows), stage)
            else:
                print(f"Get numberized rows from cache in stage: {stage}")
        else:
            numberized_rows = self.numberize_rows(indexed_rows)
        sort_key = self.sort_key

        def key(row):
            return self.tensorizers[sort_key].sort_key(row.numberized[sort_key])

        if load_early:
            batcher = Batcher(
                self.batcher.train_batch_size,
                self.batcher.eval_batch_size,
                self.batcher.test_batch_size,
            )
        else:
            batcher = self.batcher
        batches = batcher.batchify(
            numberized_rows, sort_key=(key if sort_key else None), stage=stage
        )
        return pad_and_tensorize_batches(self.tensorizers, batches)

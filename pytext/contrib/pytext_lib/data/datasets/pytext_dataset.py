#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import itertools
import logging
import random
from typing import Callable, Iterable, Optional, Union

import torch.nn as nn
from pytext.contrib.pytext_lib.data.datasets.batchers import Batcher
from pytext.contrib.pytext_lib.transforms.transforms import (
    IdentityTransform,
    RowsToColumnarTransform,
)
from pytext.contrib.pytext_lib.utils.data_util import (
    columnar_tuple_to_rows,
    rows_to_columnar_tuple,
)
from pytext.data.sources.data_source import shard
from torch.utils.data import IterableDataset


logger = logging.getLogger(__name__)


class PyTextDataset(IterableDataset):
    def __init__(
        self,
        iterable: Iterable,
        batch_size: int = 1,
        is_shuffle: bool = True,
        transform: Optional[Union[nn.Module, Callable]] = None,
        custom_batcher: Optional[Batcher] = None,
        collate_fn: Optional[Callable] = None,
        chunk_size: Optional[int] = 1000,
        is_cycle: bool = False,
        length: Optional[int] = None,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.iterable = itertools.cycle(iterable) if is_cycle else iterable
        if world_size > 1:
            logger.error(f"data sharding for rank: {rank}, world_size: {world_size}")
            self.iterable = shard(self.iterable, rank, world_size)
        self.batch(batch_size, custom_batcher)
        self.is_shuffle = is_shuffle
        self.transform = RowsToColumnarTransform(transform or IdentityTransform())
        self.collate_fn = collate_fn

        self.chunk_size = chunk_size  # num of batches per chunk
        self.is_cycle = is_cycle
        self.length = length

        if self.chunk_size and self.batch_size:
            self.iterable = ChunkIterator(
                self.iterable, self.chunk_size * self.batch_size, self.length
            )

    def __iter__(self):
        for chunk in iter(self.iterable):
            if self.is_shuffle:
                random.shuffle(chunk)
            if self.custom_batcher:
                # PoolingBatcher requires "sorting by sequence length"
                # in this case, we need to transform the chunk then batching.
                # data flow:
                #   chunk of rows -> transformed chunk -> batch -> transformed batch -> model input
                # TODO: optimize performance by batching on columnar format directly,
                # instead of converting to rows format then converting back
                # note: rows format is easy for batching, columnar format is more memory efficient
                chunk = self.transform(chunk)
                chunk = columnar_tuple_to_rows(chunk)
                for batch in self.custom_batcher.batchify(chunk):
                    batch = rows_to_columnar_tuple(batch)
                    yield self.collate_fn(*batch) if self.collate_fn else batch
            elif self.batcher:
                # for regular case, batching is done before tranform
                # data flow:
                #   chunk of rows -> batch -> transformed batch -> model input
                for batch in self.batcher.batchify(chunk):
                    batch = self.transform(batch)
                    yield self.collate_fn(*batch) if self.collate_fn else batch
            else:
                # in case no batching needed
                batch = self.transform(chunk)
                yield self.collate_fn(*batch) if self.collate_fn else batch

    def transform(self, transform):
        self.transform = transform
        return self

    def batch(self, batch_size: Optional[int] = None, custom_batcher=None):
        assert not (
            batch_size and custom_batcher
        ), "batch_size and custom_batcher are mutual exclusive"
        self.batch_size = custom_batcher.batch_size if custom_batcher else batch_size
        self.batcher = Batcher(batch_size) if batch_size else None
        self.custom_batcher = custom_batcher
        return self

    def shuffle(self, is_shuffle):
        self.is_shuffle = is_shuffle
        return self

    def collate(self, collate_fn):
        self.collate_fn = collate_fn
        return self


class ChunkIterator:
    def __init__(self, iterator, chunk_size: int, length: Optional[int]):
        self.iterator = iterator
        self.chunk_size = chunk_size
        self.length = length

    def __iter__(self):
        data = []
        for i, example in enumerate(self.iterator):
            data.append(example)
            if len(data) == self.chunk_size:
                yield data
                data = []

            # stop here because it reaches #length data
            if self.length and i >= self.length - 1:
                break

        if len(data) > 0:
            yield data
            data = []

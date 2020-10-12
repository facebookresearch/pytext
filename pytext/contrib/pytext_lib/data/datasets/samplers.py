#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import random
from typing import Callable, Optional, Sized

from torch.utils.data.sampler import BatchSampler, SequentialSampler


class SortedSampler(SequentialSampler):
    """Samples elements in sorted order, according to a given key."""

    def __init__(self, data_source: Sized, key: Callable):
        indices_with_keys = ((key(row), idx) for idx, row in enumerate(data_source))
        self.sorted_indices = [idx for _, idx in sorted(indices_with_keys)]
        super().__init__(self.sorted_indices)

    def __iter__(self):
        return iter(self.sorted_indices)


class PooledBatchSampler(BatchSampler):
    """
    Wraps an iterable to yield batches of indices from a sorted pool.

    This sampler sequentially samples a pool of `num_batches_in_pool * batch_size` items
    from the base iterable, optionally sorts them according to a given key, and batches them.
    Batches from a pool are returned in random order.

    Increasing the size of the pool (larger `num_batches_in_pool`) means that the items in
    the resulting batches will have more uniform length, but it also increases memory usage.
    """

    def __init__(
        self,
        data_source: Sized,
        batch_size: int,
        drop_last: bool,
        key: Optional[Callable] = None,
        num_batches_in_pool: int = 100,
    ):
        sampler = SequentialSampler(data_source)
        super().__init__(sampler, batch_size, drop_last)
        self.data_source = data_source
        self.key = key
        self.pool_sampler = BatchSampler(
            sampler=sampler,
            batch_size=min(batch_size * num_batches_in_pool, len(sampler)),
            drop_last=False,
        )

    def __iter__(self):
        for pool_indices in self.pool_sampler:
            if self.key is not None:
                in_pool_sampler = SortedSampler(
                    pool_indices, key=lambda i: self.key(self.data_source[i])
                )
            else:
                in_pool_sampler = SequentialSampler(pool_indices)
            batch_sampler = BatchSampler(
                in_pool_sampler, self.batch_size, self.drop_last
            )
            batches = list(batch_sampler)
            random.shuffle(batches)
            for batch_indices in batches:
                yield [pool_indices[i] for i in batch_indices]

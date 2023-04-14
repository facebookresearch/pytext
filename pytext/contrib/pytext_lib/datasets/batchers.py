#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import math
import random
from typing import Any, Iterable


class Batcher:
    """Batcher designed to batch rows of data, before padding."""

    def __init__(self, batch_size, sort_key=None):
        self.batch_size = batch_size
        self.sort_key = sort_key

    def batchify(self, iterable):
        self.iterable = iterable
        return self

    def __iter__(self):
        """
        Group rows by batch_size. The last batch will be of length
        len(iterable) % batch_size
        """
        return self._group_iter(self.iterable, self.batch_size, self.sort_key)

    def _group_iter(self, iterable: Iterable[Any], group_size: int = 1, sort_key=None):
        group = []
        for row in iterable:
            group.append(row)
            if len(group) == group_size:
                if sort_key:
                    group.sort(key=sort_key, reverse=True)
                yield group
                group = []
        # the last batch
        if len(group) > 0:
            if sort_key:
                group.sort(key=sort_key, reverse=True)
            yield group


class PoolingBatcher(Batcher):
    """
    Batcher that loads a pool of data, sorts it, and batches it.

    Shuffling is performed before pooling, by loading `num_shuffled_pools` worth
    of data, shuffling, and then splitting that up into pools.
    """

    def __init__(
        self, batch_size, pool_num_batches=1000, num_shuffled_pools=1, sort_key=None
    ):
        super().__init__(batch_size)
        assert pool_num_batches >= 1 and num_shuffled_pools >= 1
        self.pool_num_batches = pool_num_batches
        self.num_shuffled_pools = num_shuffled_pools
        self.sort_key = sort_key

    def batchify(self, iterable):
        self.iterable = iterable
        return self

    def __iter__(self):
        """
        From an iterable of dicts, yield dicts of lists:

        1. Load `num_shuffled_pools` pools of data, and shuffle them.
        2. Load a pool (`batch_size * pool_num_batches` examples).
        3. Sort rows, if necessary.
        4. Shuffle the order in which the batches are returned, if necessary.
        """
        pool_size = self.batch_size * self.pool_num_batches
        super_pool_size = pool_size * self.num_shuffled_pools

        for super_pool in self._group_iter(self.iterable, super_pool_size):
            # No point in shuffling if we're loading a single pool which is then sorted.
            if self.num_shuffled_pools > 1 or self.sort_key is None:
                random.shuffle(super_pool)
            for pool in self._group_iter(super_pool, pool_size, self.sort_key):
                batch_indices = list(range(math.ceil(len(pool) / self.batch_size)))
                if self.sort_key:
                    random.shuffle(batch_indices)
                for batch_index in batch_indices:
                    batch = pool[
                        self.batch_size
                        * batch_index : self.batch_size
                        * (batch_index + 1)
                    ]
                    yield batch

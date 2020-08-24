#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
import random
from typing import Any, List


class Batcher:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def batchify(self, examples):
        for batch in self._group(examples, self.batch_size):
            yield batch

    def _group(self, examples: List[Any], group_size: int = 1, sort_key=None):
        group = []
        for example in examples:
            group.append(example)
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
    def __init__(self, batch_size, pool_num_batches=1000, sort_key=None):
        self.batch_size = batch_size
        self.pool_num_batches = pool_num_batches
        self.sort_key = sort_key

    def batchify(self, examples):
        """
        1. Load a pool (`batch_size * pool_num_batches` rows).
        2. Sort rows, if necessary.
        3. Shuffle the order in which the batches are returned, if necessary.
        """
        pool_size = self.batch_size * self.pool_num_batches

        for pool in self._group(examples, pool_size, self.sort_key):
            batch_indices = list(range(math.ceil(len(pool) / self.batch_size)))
            if self.sort_key:
                random.shuffle(batch_indices)
            for batch_index in batch_indices:
                batch = pool[
                    self.batch_size * batch_index : self.batch_size * (batch_index + 1)
                ]
                yield batch

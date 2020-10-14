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


class PagedBatchSampler(BatchSampler):
    """
    Wraps an iterable to yield batches of indices from a sorted page.

    This sampler sequentially samples a page of `num_batches_in_page * batch_size` items
    from the base iterable, optionally sorts them according to a given key, and batches them.
    Batches from a page are returned in random order.

    Increasing the size of the page (larger `num_batches_in_page`) means that the items in
    the resulting batches will have more uniform length, but it also increases memory usage.
    """

    def __init__(
        self,
        data_source: Sized,
        batch_size: int,
        drop_last: bool,
        key: Optional[Callable] = None,
        num_batches_in_page: int = 1024,
    ):
        sampler = SequentialSampler(data_source)
        super().__init__(sampler, batch_size, drop_last)
        self.data_source = data_source
        self.key = key
        self.page_sampler = BatchSampler(
            sampler=sampler,
            batch_size=min(batch_size * num_batches_in_page, len(sampler)),
            drop_last=False,
        )

    def __iter__(self):
        for page_indices in self.page_sampler:
            if self.key is not None:
                in_page_sampler = SortedSampler(
                    page_indices, key=lambda i: self.key(self.data_source[i])
                )
            else:
                in_page_sampler = SequentialSampler(page_indices)
            batch_sampler = BatchSampler(
                in_page_sampler, self.batch_size, self.drop_last
            )
            batches = list(batch_sampler)
            random.shuffle(batches)
            for batch_indices in batches:
                yield [page_indices[i] for i in batch_indices]

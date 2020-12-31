#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import csv
import json
import logging
from typing import Callable, Iterable, Iterator, List, Optional, Union

import torch.nn as nn
from pytext.data.sources.data_source import SafeFileWrapper
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)


class PagedDataset(Dataset):
    """A map-style dataset built out of an iterable.

    Fetches `page_size` items from an iterable at a time, allowing random access.
    This removes the need to use Pytorch's `IterableDataset`s for e.g. larger-than-memory
    TSV files, which have the disadvantage of not supporting custom samplers.

    When an item is requested, the following occurs:
    * If it is *in* the currently loaded page, it's simply returned.
    * If it is *after* the currently loaded page, the next page is loaded.
    * If it is *before* the currently loaded page, a new iterator is obtained by calling
      the iterable's `__iter__()` method, and its first page is loaded.

    Given that fetching a previous page involves restarting iteration from the beginning,
    users should avoid the use of samplers which will make requests for previous page,
    unless this is at the beginning of a new epoch. Random sampling can be achieved by
    using `PagedBatchSampler` and matching its page size to the `PagedDataset`'s page size.

    NB: Users should ensure that `PagedDataset` is passed an `Iterable` which will allow
    repeated passes over its members, and not an `Iterator`, which will be consumed after
    the first epoch.
    """

    def __init__(
        self,
        iterable: Iterable,
        total_size: Optional[int] = None,
        page_size: int = 1024,
        transform: Optional[Union[nn.Module, Callable]] = None,
    ):
        self.iterable = iterable
        assert not isinstance(
            iterable, Iterator
        ), "PagedDataset was passed an Iterator, which doesn't allow repeated passes."
        if total_size is None:
            self.total_size = sum(1 for _ in self.iterable)
        else:
            self.total_size = total_size
        self.page_size = page_size
        self.transform = transform

        self.current_page = []
        self._init_iterator()
        self._get_next_page()

    def __getitem__(self, idx):
        if idx >= self.total_size or idx < 0:
            raise IndexError(
                f"Index #{idx} is out of bounds for dataset of size {self.total_size}."
            )

        # Compute the index of the requested item relative to the start of the current page.
        relative_idx = idx - self.current_page_num * self.page_size
        logger.debug(f"Requested index #{idx} has relative index #{relative_idx}.")

        if relative_idx < 0:  # requested item is before the current page
            logger.debug("Re-initializing iterator of PagedDataset")
            self._init_iterator()
            self._get_next_page()
            return self[idx]
        if relative_idx < len(self.current_page):
            return self.current_page[relative_idx]
        else:
            self._get_next_page()
            return self[idx]

    def __len__(self):
        return self.total_size

    def _get_next_page(self):
        if self.last_page:
            raise ValueError("Attempted to go beyond the last dataset page.")

        self.current_page_num += 1
        self.current_page = []
        for _ in range(self.page_size):
            try:
                item = next(self.iterator)
                if self.transform:
                    item = self.transform(item)
                self.current_page.append(item)
            except StopIteration:
                self.is_last_page = True
                break

        logger.debug(
            f"Loaded next page (#{self.current_page_num}) with size {len(self.current_page)}."
        )

    def _init_iterator(self):
        self.iterator = iter(self.iterable)
        self.current_page_num = -1
        self.last_page = False


class TsvDataset(PagedDataset):
    def __init__(
        self,
        path: str,
        column_names: Optional[List[str]] = None,
        delimiter: str = "\t",
        **kwargs,
    ):
        self.path = path
        self.column_names = column_names
        self.delimiter = delimiter
        super().__init__(self, **kwargs)

    def __iter__(self):
        logger.debug(f"Initializing TSV iterator for {self.path}.")
        file = SafeFileWrapper(self.path, encoding="utf-8")
        yield from csv.DictReader(
            file, delimiter=self.delimiter, fieldnames=self.column_names
        )


class JsonlDataset(PagedDataset):
    def __init__(self, path: str, **kwargs):
        self.path = path
        super().__init__(self, **kwargs)

    def __iter__(self):
        logger.debug(f"Initializing JSONL iterator for {self.path}.")
        file = SafeFileWrapper(self.path, encoding="utf-8")
        for line in file:
            yield json.loads(line)

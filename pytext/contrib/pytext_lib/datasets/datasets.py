#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import csv
import sys
import threading
from typing import Any, Dict, List

import torch.nn as nn
from pytext.contrib.pytext_lib.datasets import Batcher
from pytext.contrib.pytext_lib.transforms import (
    DictToListTransform,
    ListToDictTransform,
)
from pytext.data.sources.data_source import SafeFileWrapper
from torch.utils.data import IterableDataset


class TSV:
    def __init__(
        self,
        file,
        field_names=None,
        delimiter="\t",
        quoted=False,
        drop_incomplete_rows=False,
    ):
        self.file = file
        self.field_names = field_names
        self.delimiter = delimiter
        self.quoted = quoted
        self.drop_incomplete_rows = drop_incomplete_rows
        self.total_rows_count = 0
        self.incomplete_rows_count = 0
        self._access_lock = threading.Lock()
        csv.field_size_limit(sys.maxsize)

    def __iter__(self):
        can_acquire = self._access_lock.acquire(blocking=False)
        if not can_acquire:
            raise Exception("Concurrent iteration not supported")
        self.file.seek(0)
        try:
            reader = csv.DictReader(
                (line.replace("\0", "") for line in self.file),
                fieldnames=self.field_names,
                delimiter=self.delimiter,
                quoting=csv.QUOTE_MINIMAL if self.quoted else csv.QUOTE_NONE,
            )
            if self.drop_incomplete_rows:
                for row in reader:
                    self.total_rows_count += 1
                    if any(map(lambda v: v is None, row.values())):  # drop!
                        self.incomplete_rows_count += 1
                        continue
                    yield row
            else:
                yield from reader
        finally:
            self._access_lock.release()


class TsvDataset(IterableDataset):
    def __init__(
        self,
        file_path: str,
        field_names: List[str],
        batcher: Batcher = None,
        batch_size: int = 1,
        transform: nn.Module = None,
        label_transform: nn.Module = None,
        delimiter: str = "\t",
    ):
        super().__init__()
        self.transform = transform or ListToDictTransform()
        self.label_transform = label_transform or ListToDictTransform()

        self.file_path = file_path
        self.file = SafeFileWrapper(self.file_path, encoding="utf-8", errors="replace")
        iterator = TSV(self.file, field_names=field_names, delimiter=delimiter)
        batcher = batcher or Batcher(batch_size)
        self.iterator = batcher.batchify(iterator)

    def __iter__(self) -> Dict[str, List[Any]]:
        for batch in self.iterator:
            input_dict = self.transform(self.transform.extract_inputs(batch))
            label_dict = self.label_transform(batch)
            yield {**input_dict, **label_dict}


def transpose_dataset(dataset: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    dict_to_list_transform = DictToListTransform()
    for batch in dataset:
        # convert dict_of_list to list_of_dict
        list_of_dict = dict_to_list_transform(batch)
        if len(list_of_dict) == 1:
            list_of_dict = list_of_dict[0]
        yield list_of_dict


class NestedDataset(IterableDataset):
    def __init__(
        self,
        dataset: Dict[str, List[Any]],
        batcher: Batcher = None,
        batch_size: int = 1,
        transform: nn.Module = None,
        label_transform: nn.Module = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transform or ListToDictTransform()
        self.label_transform = label_transform or ListToDictTransform()

        iterator = transpose_dataset(dataset)
        batcher = batcher or Batcher(batch_size)
        self.iterator = batcher.batchify(iterator)

    def __iter__(self) -> Dict[str, List[Any]]:
        for batch in self.iterator:
            input_dict = self.transform(self.transform.extract_inputs(batch))
            label_dict = self.label_transform(batch)
            yield {**input_dict, **label_dict}

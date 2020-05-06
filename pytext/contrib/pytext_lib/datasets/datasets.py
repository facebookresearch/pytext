#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import csv
import sys
import threading

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
        file_path,
        field_names,
        batch_size=1,
        transform=None,
        label_transform=None,
        delimiter="\t",
    ):
        super().__init__()
        self.file_path = file_path
        self.batch_size = batch_size
        self.transform = transform
        self.label_transform = label_transform

        self.file = SafeFileWrapper(self.file_path, encoding="utf-8", errors="replace")
        self.iterator = TSV(self.file, field_names=field_names, delimiter=delimiter)

    def __iter__(self):
        batch = []
        tensors = {}
        for example in self.iterator:
            # if batch_size is None:
            #     yield self.transform(self.transform.extract_inputs(batch))
            # else:
            batch.append(example)
            if len(batch) == self.batch_size:
                if self.transform:
                    tensors.update(self.transform(self.transform.extract_inputs(batch)))
                else:
                    tensors.update(batch)

                if self.label_transform:
                    tensors.update(self.label_transform(batch))
                yield tensors

                tensors.clear()
                batch.clear()


class NestedDataset(IterableDataset):
    def __init__(self, dataset, batch_size=1, transform=None, label_transform=None):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transform
        self.label_transform = label_transform
        self.iterator = dataset

    def __iter__(self):
        batch = []
        tensors = {}
        for example in self.iterator:
            batch.append(example)
            if len(batch) == self.batch_size:
                tensors.update(self.transform(self.transform.extract_inputs(batch)))
                tensors.update(self.label_transform(batch))
                yield tensors

                tensors.clear()
                batch.clear()

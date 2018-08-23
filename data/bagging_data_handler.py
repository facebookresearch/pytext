#!/usr/bin/env python3
from .data_handler import DataHandler
from torchtext import data as textdata

from typing import Tuple
import random
from pytext.utils import cuda_utils


class BaggingDataHandler(DataHandler):
    def __init__(self, data_fraction: float, *args, **kwargs) -> None:
        super(BaggingDataHandler, self).__init__(*args, **kwargs)
        self.data_fraction = data_fraction

    def batch(
        self, file_names: Tuple[str, ...], batch_size: Tuple[int, ...]
    ) -> Tuple[textdata.Iterator, ...]:
        return textdata.BucketIterator.splits(
            tuple(
                sample(self.gen_dataset_from_file(f), self.data_fraction)
                for f in file_names
            ),
            batch_sizes=batch_size,
            device=None if cuda_utils.CUDA_ENABLED else -1,
            sort_within_batch=True,
            repeat=False,
        )


def sample(dataset: textdata.Dataset, data_fraction: float) -> textdata.Dataset:
    number_of_samples = max(int(len(dataset) * data_fraction), 1, len(dataset))
    return textdata.Dataset(
        examples=random.sample(dataset.examples, number_of_samples),
        fields=dataset.fields,
        preprocess_workers=dataset.preprocess_workers,
    )

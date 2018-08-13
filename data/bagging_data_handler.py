#!/usr/bin/env python3
from .data_handler import DataHandler
from torchtext import data as textdata

from typing import Tuple
import random
from pytext.utils import cuda_utils


class BaggingDataHandler(DataHandler):
    def __init__(self, data_fraction: float, *arg, **kwarg) -> None:
        super(BaggingDataHandler, self).__init__(*arg, **kwarg)
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
    desired_n_rows = int(len(dataset) * data_fraction)
    sampled_indices = random.sample(range(len(dataset)), desired_n_rows)
    sampled_examples = [dataset.examples[i] for i in sampled_indices]
    return textdata.Dataset(
        examples=sampled_examples,
        fields=dataset.fields,
        preprocess_workers=dataset.preprocess_workers,
    )

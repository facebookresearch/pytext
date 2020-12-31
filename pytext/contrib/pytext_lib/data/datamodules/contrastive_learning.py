#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
from random import sample

import pytorch_lightning as pl
from pytext.contrib.pytext_lib.data.datasets import JsonlDataset, PagedBatchSampler
from pytext.contrib.pytext_lib.transforms import ModelTransform
from torch.utils.data import DataLoader


logger = logging.getLogger(__name__)


class ContrastiveLearningDataModule(pl.LightningDataModule):
    def __init__(
        self,
        transform: ModelTransform,
        train_path,
        batch_size: int,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.doc_transform = transform
        self.batch_size = batch_size

        self.train_dataset = JsonlDataset(
            train_path,
            transform=self._transform,
        )

    def _transform(self, item):
        anchor = self.doc_transform.transform([item["anchor"]])
        positive = self.doc_transform.transform([sample(item["positives"], 1)[0]])
        return {"anchor": anchor, "positive": positive}

    def _collate_fn(self, batch):
        return {
            "anchor": self.doc_transform.collate_fn(
                *self._merge_batches(batch, "anchor")
            ),
            "positive": self.doc_transform.collate_fn(
                *self._merge_batches(batch, "positive")
            ),
        }

    def _merge_batches(self, batch, key):
        result = []
        merged_columns = list(zip(*[x[key] for x in batch]))
        for columns in merged_columns[:4]:
            column = []
            for row in columns:
                column.extend(row)
            result.append(column)
        return result

    def train_dataloader(self):
        sampler = PagedBatchSampler(
            data_source=self.train_dataset,
            batch_size=self.batch_size,
            drop_last=True,
        )
        return DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            collate_fn=self._collate_fn,
        )

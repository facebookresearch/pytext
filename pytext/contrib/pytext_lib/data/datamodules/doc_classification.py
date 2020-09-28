#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
from functools import partial
from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
import torch
from pytext.contrib.pytext_lib.data.datasets import TsvDataset
from pytext.contrib.pytext_lib.transforms import ModelTransform
from torch.utils.data import DataLoader


logger = logging.getLogger(__name__)


class DocClassificationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        transform: ModelTransform,
        # Dataset args
        train_path: str,
        val_path: str,
        test_path: str,
        columns: List[Any] = None,
        column_mapping: Optional[Dict[str, str]] = None,
        delimiter: str = "\t",
        batch_size: Optional[int] = None,
        is_shuffle: bool = True,
        chunk_size: int = 1000,
        is_cycle: bool = False,
        length: Optional[int] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.transform = transform
        self.dataset_partial = partial(
            TsvDataset,
            columns=columns,
            column_mapping=column_mapping,
            delimiter=delimiter,
            batch_size=batch_size,
            is_shuffle=is_shuffle,
            transform=self.transform.transform,
            collate_fn=self.transform.collate_fn,
            chunk_size=chunk_size,
            is_cycle=is_cycle,
            length=length,
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=None)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=None)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=None)

    def setup(self, stage: str):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            rank = (
                torch.distributed.get_rank()
                if torch.distributed.is_initialized()
                else 0
            )
            logger.debug(f"setup for rank: {self.rank}, world_size: {self.world_size}")
        else:
            world_size = 1
            rank = 0
        datasets = [
            self.dataset_partial(path=path, rank=rank, world_size=world_size)
            for path in (self.train_path, self.val_path, self.test_path)
        ]
        self.train_dataset, self.val_dataset, self.test_dataset = datasets

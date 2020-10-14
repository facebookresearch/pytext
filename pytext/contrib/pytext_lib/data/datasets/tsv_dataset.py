#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
from typing import Any, Callable, Dict, List, Optional

import torch.nn as nn
from pytext.contrib.pytext_lib.data.datasets.batchers import Batcher
from pytext.contrib.pytext_lib.data.datasets.pytext_dataset import PyTextDataset
from pytext.data.sources.data_source import SafeFileWrapper
from pytext.data.sources.tsv import TSV


logger = logging.getLogger(__name__)


class DeprecatedTsvDataset(PyTextDataset):
    def __init__(
        self,
        path: str,
        columns: List[Any] = None,
        column_mapping: Optional[Dict[str, str]] = None,
        delimiter: str = "\t",
        batch_size: Optional[int] = None,
        is_shuffle: bool = True,
        transform: Optional[nn.Module] = None,
        custom_batcher: Optional[Batcher] = None,
        collate_fn: Optional[Callable] = None,
        chunk_size: int = 1000,
        is_cycle: bool = False,
        length: Optional[int] = None,
        rank: int = 0,
        world_size: int = 1,
        *args,
        **kwargs,
    ):
        logger.debug(f"init TsvDataset from: {path}")
        columns = columns or ["text", "label"]
        if column_mapping:
            raise NotImplementedError("column mapping is not supported for tsv yet!")
        self.file = SafeFileWrapper(path, encoding="utf-8", errors="replace")
        tsv_iterator = TSV(self.file, field_names=columns, delimiter=delimiter)
        super().__init__(
            iterable=tsv_iterator,
            batch_size=batch_size,
            is_shuffle=is_shuffle,
            transform=transform,
            custom_batcher=custom_batcher,
            collate_fn=collate_fn,
            chunk_size=chunk_size,
            is_cycle=is_cycle,
            length=length,
            rank=rank,
            world_size=world_size,
        )

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import multiprocessing
from typing import NamedTuple, Optional

from .constants import (
    BatchContext,
    DatasetFieldName,
    DFColumn,
    PackageFileName,
    Padding,
    Stage,
    VocabMeta,
)


class QueueChannel(NamedTuple):
    # subprocess send trainer logging back to main process
    logging_queue: Optional[multiprocessing.Queue] = None
    # subprocess callback back to main process
    # for example: start fine tuning task based on current snapshot
    callback_queue: Optional[multiprocessing.Queue] = None

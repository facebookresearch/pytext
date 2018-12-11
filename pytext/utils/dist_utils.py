#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import signal

import torch
import torch.distributed as dist_c10d


def dist_init(
    distributed_rank: int, world_size: int, init_method: str, backend: str = "nccl"
):
    if init_method and world_size > 1 and torch.cuda.is_available():
        dist_c10d.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=distributed_rank,
        )

        if distributed_rank != 0:
            suppress_output()


def suppress_output():
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        # force print the result when kwargs contains force and value is True
        if kwargs.pop("force", False):
            builtin_print(*args, **kwargs)

    __builtin__.print = print

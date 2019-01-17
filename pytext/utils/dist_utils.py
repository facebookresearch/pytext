#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy
import math
import os
import signal
from typing import Any, List

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


def get_shard_range(data_size: int, rank: int, world_size: int):
    """
    Add extra 1 examples in the remainder(e.g data_size % world_size) rank,
    For example, world_size = 8, data_size = 66
    The shard size is [9, 9, 8, 8, 8, 8, 8, 8]
    """
    remainder = data_size % world_size
    shard_len = data_size // world_size
    if rank < remainder:
        shard_len += 1

    shard_offset = rank * shard_len + min(rank, remainder)
    shard_end = data_size if rank == world_size - 1 else shard_offset + shard_len
    return (shard_offset, shard_end)


def pad_shard_data(shard_data: List[Any], data_size: int, world_size: int):
    """
    Because of the trailing data (dataset_size % world_size), some shard could
    have 1 less example than the maximum shard, in this case we will pad to
    ensure every shard have the same size.
    The impact should be negligible when dataset_size >> world_size
    """
    # TODO: the workaround is that we could store max_shard_size in Dataset

    max_shard_size = math.ceil(data_size / float(world_size))
    shard_data_size = len(shard_data)

    if shard_data_size == max_shard_size:
        pass
    elif shard_data_size == max_shard_size - 1:
        shard_data.append(copy.deepcopy(shard_data[-1]))
    else:
        raise ValueError(
            f"shard_data_size should equal or one less than max_shard_size, "
            + f"shard_data_size is {shard_data_size} and max_shard_size "
            + f"{max_shard_size}"
        )

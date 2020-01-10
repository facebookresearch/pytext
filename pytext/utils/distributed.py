#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import pytext.utils.cuda as cuda
import torch
import torch.distributed as dist_c10d


_round_robin_process_group = None


def dist_init(
    distributed_rank: int,
    world_size: int,
    init_method: str,
    device_id: int,
    backend: str = "nccl",
    gpu_streams: int = 1,
):
    """
    1. After spawn process per GPU, we want all workers to call init_process_group
    around the same time or times out.
    2. After dist_init, we want all workers to start calling all_reduce/barrier
    around the same time or NCCL timeouts.
    """
    global _round_robin_process_group

    if init_method and world_size > 1 and torch.cuda.is_available():
        dist_c10d.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=distributed_rank,
        )
        # calling all_reduce for synchronzing all workers
        dist_tensor = torch.tensor(
            [1], dtype=torch.float32, device="cuda:{}".format(device_id)
        )
        dist_c10d.all_reduce(dist_tensor)

        if gpu_streams >= 1:
            _round_robin_process_group = dist_c10d._round_robin_process_groups(
                [dist_c10d.new_group(backend=backend) for _ in range(gpu_streams)]
            )

            for _ in range(gpu_streams):
                dist_tensor = torch.tensor(
                    [1], dtype=torch.float32, device="cuda:{}".format(device_id)
                )
                _round_robin_process_group.allreduce(dist_tensor)
            print(f"Using {gpu_streams} GPU streams for gradient sync.")

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


def force_print(*args, **kwargs):
    if cuda.CUDA_ENABLED and cuda.DISTRIBUTED_WORLD_SIZE > 1:
        try:
            device_info = f" [device:{torch.cuda.current_device()}]"
            print(*args, device_info, **kwargs, force=True)
        except TypeError:
            pass
    else:
        print(*args, **kwargs)


def get_shard_range(dataset_size: int, rank: int, world_size: int):
    """
    In case dataset_size is not evenly divided by world_size, we need to pad
    one extra example in each shard
    shard_len = dataset_size // world_size + 1

    Case 1 rank < remainder: each shard start position is rank * shard_len

    Case 2 rank >= remainder: without padding, each shard start position is
    rank * (shard_len - 1) + remainder = rank * shard_len - (rank - remainder)
    But to make sure all shard have same size, we need to pad one extra example
    when rank >= remainder, so start_position = start_position - 1

    For example, dataset_size = 21, world_size = 8
    rank 0 to 4: [0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]
    rank 5 to 7: [14, 15, 16], [16, 17, 18], [18, 19, 20]
    """
    remainder = dataset_size % world_size
    shard_len = dataset_size // world_size

    if remainder == 0:
        shard_offset = rank * shard_len
    else:
        # take one extra when dataset_size is not evenly divided by world_size
        shard_len += 1
        shard_offset = rank * shard_len - max(0, rank + 1 - remainder)
    shard_end = shard_offset + shard_len - 1

    return (shard_offset, shard_end)

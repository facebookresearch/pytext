#!/usr/bin/env python3

import tempfile

import torch
from pytext.config import PyTextConfig, TestConfig
from pytext.utils.dist_utils import ErrorHandler
from pytext.workflow import test_model, train_model

from .args import TEST_MODE, parse_config
from .serialize import config_from_json, config_to_json


def run_job():
    mode, config = parse_config()
    if mode == TEST_MODE:
        print("Start testing...")
        test_model(config)
    else:
        print("Starting training...")
        train_model_distributed(config)
        print("Starting testing...")
        test_config = TestConfig(
            load_snapshot_path=config.save_snapshot_path,
            test_path=config.jobspec.data_handler.test_path,
            use_cuda_if_available=config.use_cuda_if_available,
        )
        test_model(test_config)


def train_model_distributed(config):
    assert (
        config.use_cuda_if_available and torch.cuda.is_available()
    ) or config.distributed_world_size == 1, (
        "distributed training is only available for GPU training"
    )
    assert (
        config.distributed_world_size == 1
        or config.distributed_world_size <= torch.cuda.device_count()
    ), "Only {} GPUs are available, {} GPUs were requested".format(
        torch.cuda.device_count(), config.distributed_world_size
    )

    print("Starting training, World size is {}".format(config.distributed_world_size))
    procs = []
    mp = torch.multiprocessing.get_context("spawn")
    error_queue = mp.SimpleQueue()
    error_handler = ErrorHandler(error_queue)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".dist_sync") as sync_file:
        dist_init_method = "file://" + sync_file.name

        for i in range(config.distributed_world_size):
            distributed_rank = i
            device_id = i
            procs.append(
                mp.Process(
                    target=run_single,
                    args=(
                        config_to_json(PyTextConfig, config),
                        device_id,
                        distributed_rank,
                        config.distributed_world_size,
                        dist_init_method,
                    ),
                    daemon=True,
                )
            )
            procs[i].start()
            error_handler.add_child(procs[i].pid)
        for p in procs:
            p.join()


def run_single(config_json, device_id, rank, world_size, dist_init_method):
    config = config_from_json(PyTextConfig, config_json)
    train_model(config, dist_init_method, device_id, rank, world_size)

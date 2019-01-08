#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
from typing import Any, Dict, List, Optional, Tuple, get_type_hints

import torch
from pytext.config import PyTextConfig, TestConfig
from pytext.config.component import create_exporter
from pytext.data.data_handler import CommonMetadata
from pytext.metric_reporters.channel import TensorBoardChannel
from pytext.task import Task, create_task, load, save
from pytext.utils.dist_utils import dist_init
from tensorboardX import SummaryWriter

from .utils import cuda_utils


def _set_cuda(
    use_cuda_if_available: bool, device_id: int = 0, world_size: int = 1
) -> None:
    cuda_utils.CUDA_ENABLED = use_cuda_if_available and torch.cuda.is_available()
    cuda_utils.DISTRIBUTED_WORLD_SIZE = world_size

    if use_cuda_if_available and not cuda_utils.CUDA_ENABLED:
        print("Cuda is not available, running on CPU...")
    elif cuda_utils.CUDA_ENABLED:
        torch.cuda.set_device(device_id)

    print(
        """
    # for debug of GPU
    use_cuda_if_available: {}
    device_id: {}
    world_size: {}
    torch.cuda.is_available(): {}
    cuda_utils.CUDA_ENABLED: {}
    cuda_utils.DISTRIBUTED_WORLD_SIZE: {}
    """.format(
            use_cuda_if_available,
            device_id,
            world_size,
            torch.cuda.is_available(),
            cuda_utils.CUDA_ENABLED,
            cuda_utils.DISTRIBUTED_WORLD_SIZE,
        )
    )


def prepare_task_metadata(config: PyTextConfig) -> CommonMetadata:
    """
    Loading the whole dataset into cpu memory on every single processes could
    cause OOMs for data parallel distributed training.
    To avoid such practice, we move the operations that required loading the
    whole dataset out of spawn, and pass the context to every single process.
    """
    return create_task(config.task).data_handler.metadata


def train_model(
    config: PyTextConfig,
    dist_init_url: str = None,
    device_id: int = 0,
    rank: int = 0,
    world_size: int = 1,
    summary_writer: Optional[SummaryWriter] = None,
    metadata: CommonMetadata = None,
) -> Tuple:
    task = prepare_task(
        config, dist_init_url, device_id, rank, world_size, summary_writer, metadata
    )
    trained_model, best_metric = task.train(config, rank, world_size)
    # Only rank 0 gets to finalize the job and export the model
    if rank == 0:
        save_and_export(config, task, summary_writer)
    return trained_model, best_metric


def prepare_task(
    config: PyTextConfig,
    dist_init_url: str = None,
    device_id: int = 0,
    rank: int = 0,
    world_size: int = 1,
    summary_writer: Optional[SummaryWriter] = None,
    metadata: CommonMetadata = None,
) -> Task:

    if dist_init_url and world_size > 1:
        assert metadata is not None
        dist_init(rank, world_size, dist_init_url)

    print("\nParameters: {}\n".format(config))
    _set_cuda(config.use_cuda_if_available, device_id, world_size)
    if config.load_snapshot_path and os.path.isfile(config.load_snapshot_path):
        task = load(config.load_snapshot_path)
    else:
        task = create_task(config.task, metadata=metadata)

    if summary_writer:
        task.metric_reporter.add_channel(
            TensorBoardChannel(summary_writer=summary_writer)
        )

    return task


def save_and_export(
    config: PyTextConfig, task: Task, summary_writer: Optional[SummaryWriter] = None
) -> None:
    print("\n=== Saving model to: " + config.save_snapshot_path)
    save(config, task.model, task.data_handler.metadata_to_save())
    task.export(
        task.model, config.export_caffe2_path, summary_writer, config.export_onnx_path
    )


def export_saved_model_to_caffe2(
    saved_model_path: str, export_caffe2_path: str, output_onnx_path: str = None
) -> None:
    task, train_config = load(saved_model_path)
    if task.exporter is None:
        TaskType = type(train_config.task)
        ExporterConfigType = get_type_hints(TaskType)["exporter"].__args__[0]
        task.exporter = create_exporter(
            ExporterConfigType(),
            train_config.task.features,
            train_config.task.labels,
            task.data_handler.metadata,
        )
    task.export(task.model, export_caffe2_path, export_onnx_path=output_onnx_path)


def test_model(
    test_config: TestConfig, summary_writer: Optional[SummaryWriter] = None
) -> Any:
    return test_model_from_snapshot_path(
        test_config.load_snapshot_path,
        test_config.use_cuda_if_available,
        test_config.test_path,
        summary_writer,
    )


def test_model_from_snapshot_path(
    snapshot_path: str,
    use_cuda_if_available: bool,
    test_path: Optional[str] = None,
    summary_writer: Optional[SummaryWriter] = None,
):
    _set_cuda(use_cuda_if_available)
    task, train_config = load(snapshot_path)
    if not test_path:
        test_path = train_config.task.data_handler.test_path
    if summary_writer:
        task.metric_reporter.add_channel(
            TensorBoardChannel(summary_writer=summary_writer)
        )

    return (task.test(test_path), train_config.task.metric_reporter.output_path)


def batch_predict(model_file: str, examples: List[Dict[str, Any]]):
    task, train_config = load(model_file)
    return task.predict(examples)

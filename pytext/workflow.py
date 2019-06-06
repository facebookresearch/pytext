#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import json
import os
from typing import Any, Dict, List, Optional, Tuple, get_type_hints

import torch
from pytext.common.constants import Stage
from pytext.config import PyTextConfig, TestConfig
from pytext.config.component import ComponentType, create_component, create_exporter
from pytext.data.data import Batcher
from pytext.data.data_handler import CommonMetadata
from pytext.metric_reporters.channel import Channel
from pytext.task import NewTask, Task_Deprecated, create_task, load, save
from pytext.task.disjoint_multitask import NewDisjointMultitask
from pytext.utils import cuda, distributed, precision, set_random_seeds, timing


def _set_cuda(
    use_cuda_if_available: bool, device_id: int = 0, world_size: int = 1
) -> None:
    cuda.CUDA_ENABLED = use_cuda_if_available and torch.cuda.is_available()
    cuda.DISTRIBUTED_WORLD_SIZE = world_size

    if use_cuda_if_available and not cuda.CUDA_ENABLED:
        print("Cuda is not available, running on CPU...")
    elif cuda.CUDA_ENABLED:
        torch.cuda.set_device(device_id)

    print(
        """
    # for debug of GPU
    use_cuda_if_available: {}
    device_id: {}
    world_size: {}
    torch.cuda.is_available(): {}
    cuda.CUDA_ENABLED: {}
    cuda.DISTRIBUTED_WORLD_SIZE: {}
    """.format(
            use_cuda_if_available,
            device_id,
            world_size,
            torch.cuda.is_available(),
            cuda.CUDA_ENABLED,
            cuda.DISTRIBUTED_WORLD_SIZE,
        )
    )


def _set_fp16(use_fp16: bool) -> None:
    # only support single GPU training at this moment.
    precision.set_fp16(fp16_enabled=use_fp16)
    print(f"# for debug of FP16: fp16_enabled={precision._FP16_ENABLED}")


def _set_distributed(
    rank: int,
    world_size: int,
    dist_init_url: str,
    device_id: int,
    metadata: CommonMetadata,
) -> None:
    if dist_init_url and world_size > 1:
        assert metadata is not None
        distributed.dist_init(rank, world_size, dist_init_url, device_id)


def prepare_task_metadata(config: PyTextConfig) -> CommonMetadata:
    """
    Loading the whole dataset into cpu memory on every single processes could
    cause OOMs for data parallel distributed training.
    To avoid such practice, we move the operations that required loading the
    whole dataset out of spawn, and pass the context to every single process.
    """
    return (
        create_task(config.task).data_handler.metadata
        if hasattr(config.task, "data_handler")
        else {}
    )


def train_model(
    config: PyTextConfig,
    dist_init_url: str = None,
    device_id: int = 0,
    rank: int = 0,
    world_size: int = 1,
    metric_channels: Optional[List[Channel]] = None,
    metadata: CommonMetadata = None,
) -> Tuple:
    task = prepare_task(
        config, dist_init_url, device_id, rank, world_size, metric_channels, metadata
    )
    trained_model, best_metric = task.train(config, rank, world_size)
    # Only rank 0 gets to finalize the job and export the model
    if rank == 0:
        save_and_export(config, task, metric_channels)
    print("Training timings")
    timing.report()
    return trained_model, best_metric


def prepare_task(
    config: PyTextConfig,
    dist_init_url: str = None,
    device_id: int = 0,
    rank: int = 0,
    world_size: int = 1,
    metric_channels: Optional[List[Channel]] = None,
    metadata: CommonMetadata = None,
) -> Task_Deprecated:

    print("\nParameters: {}\n".format(config))
    _set_cuda(config.use_cuda_if_available, device_id, world_size)
    _set_fp16(config.use_fp16)
    _set_distributed(rank, world_size, dist_init_url, device_id, metadata)

    if config.random_seed is not None:
        set_random_seeds(config.random_seed, config.use_deterministic_cudnn)

    if config.load_snapshot_path and os.path.isfile(config.load_snapshot_path):
        task, _ = load(config.load_snapshot_path)
    else:
        task = create_task(
            config.task, metadata=metadata, rank=rank, world_size=world_size
        )

    for mc in metric_channels or []:
        task.metric_reporter.add_channel(mc)

    return task


def save_and_export(
    config: PyTextConfig,
    task: Task_Deprecated,
    metric_channels: Optional[List[Channel]] = None,
) -> None:
    print("\n=== Saving model to: " + config.save_snapshot_path)
    meta = None
    if hasattr(task, "data_handler"):
        meta = task.data_handler.metadata_to_save()
    save(config, task.model, meta)
    if config.export_caffe2_path:
        task.export(
            task.model,
            config.export_caffe2_path,
            metric_channels,
            config.export_onnx_path,
        )
    if config.export_torchscript_path:
        task.torchscript_export(task.model, config.export_torchscript_path)


def export_saved_model_to_caffe2(
    saved_model_path: str, export_caffe2_path: str, output_onnx_path: str = None
) -> None:
    task, train_config = load(saved_model_path)
    if hasattr(task, "exporter") and task.exporter is None:
        TaskType = type(train_config.task)
        ExporterConfigType = get_type_hints(TaskType)["exporter"].__args__[0]
        task.exporter = create_exporter(
            ExporterConfigType(),
            train_config.task.features,
            train_config.task.labels,
            task.data_handler.metadata,
        )
    task.export(task.model, export_caffe2_path, export_onnx_path=output_onnx_path)


def export_saved_model_to_torchscript(saved_model_path: str, path: str) -> None:
    task, train_config = load(saved_model_path)
    task.torchscript_export(task.model, path)


def test_model(
    test_config: TestConfig,
    metric_channels: Optional[List[Channel]],
    test_out_path: str,
) -> Any:
    return test_model_from_snapshot_path(
        test_config.load_snapshot_path,
        test_config.use_cuda_if_available,
        test_config.test_path,
        metric_channels,
        test_out_path,
        test_config.field_names,
    )


def test_model_from_snapshot_path(
    snapshot_path: str,
    use_cuda_if_available: bool,
    test_path: Optional[str] = None,
    metric_channels: Optional[List[Channel]] = None,
    test_out_path: str = "",
    field_names: Optional[List[str]] = None,
):
    _set_cuda(use_cuda_if_available)
    task, train_config = load(snapshot_path)

    for mc in metric_channels or []:
        task.metric_reporter.add_channel(mc)

    # Overwrite the test output path because you might not have permission to
    # write to the original test output path that was created when model was trained.
    if test_out_path:
        if hasattr(task.metric_reporter, "output_path"):
            task.metric_reporter.output_path = test_out_path
        for channel in task.metric_reporter.channels:
            if hasattr(channel, "file_path"):
                channel.file_path = test_out_path
    else:
        test_out_path = train_config.task.metric_reporter.output_path

    if isinstance(task, (NewTask, NewDisjointMultitask)):
        data_source = _get_data_source(
            test_path,
            getattr(train_config.task.data, "source", None),
            field_names,
            task,
        )
        test_results = task.test(data_source)
    else:
        if not test_path:
            test_path = train_config.task.data_handler.test_path
        test_results = task.test(test_path)
    return test_results, test_out_path, metric_channels


def _get_data_source(test_path, source_config, field_names, task):
    if isinstance(task, NewDisjointMultitask):
        # Cannot easily specify a single data source for multitask
        assert not test_path
        data_source = None
    elif test_path and hasattr(source_config, "test_filename"):
        source_config.test_filename = test_path
        if field_names and hasattr(source_config, "field_names"):
            source_config.field_names = field_names
        data_source = create_component(
            ComponentType.DATA_SOURCE, source_config, task.data.data_source.schema
        )
    else:
        data_source = task.data.data_source
    return data_source


def get_logits(
    snapshot_path: str,
    use_cuda_if_available: bool,
    output_path: Optional[str] = None,
    test_path: Optional[str] = None,
    field_names: Optional[List[str]] = None,
):
    _set_cuda(use_cuda_if_available)
    task, train_config = load(snapshot_path)
    print(f"Successfully loaded model from {snapshot_path}")
    print(f"Model on GPU? {next(task.model.parameters()).is_cuda}")
    if isinstance(task, NewTask):
        task.model.eval()
        data_source = _get_data_source(
            test_path,
            getattr(train_config.task.data, "source", None),
            field_names,
            task,
        )
        task.data.batcher = Batcher()
        task.data.sort_key = None
        batches = task.data.batches(Stage.TEST, data_source=data_source)

        with open(output_path, "w", encoding="utf-8") as fout, torch.no_grad():
            for (_, tensor_dict) in batches:
                model_inputs = task.model.arrange_model_inputs(tensor_dict)
                model_outputs = task.model(*model_inputs)
                if isinstance(model_outputs, tuple):
                    model_outputs_list = [m.tolist() for m in model_outputs]
                    for row in zip(*model_outputs_list):
                        # row is a tuple of lists
                        dump_row = "\t".join(json.dumps(r) for r in row)
                        fout.write(f"{dump_row}\n")
                elif isinstance(model_outputs, torch.Tensor):
                    model_outputs_list = model_outputs.tolist()
                    for row in zip(model_outputs_list):
                        fout.write(f"{json.dumps(row)}\n")
                else:
                    raise Exception(
                        "Expecting tuple or torchTensor types for model_outputs"
                    )


def batch_predict(model_file: str, examples: List[Dict[str, Any]]):
    task, train_config = load(model_file)
    return task.predict(examples)

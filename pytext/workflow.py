#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import gzip
import json
import os
from typing import IO, Any, Dict, Iterator, List, Optional, Tuple, Union, get_type_hints

import torch
from pytext.common.constants import Stage
from pytext.config import PyTextConfig, TestConfig, ExportConfig
from pytext.config.component import ComponentType, create_component, create_exporter
from pytext.data.data import Batcher
from pytext.data.data_handler import CommonMetadata
from pytext.metric_reporters.channel import Channel
from pytext.task import (
    NewTask,
    Task_Deprecated,
    create_task,
    get_latest_checkpoint_path,
    load,
    save,
)
from pytext.task.disjoint_multitask import NewDisjointMultitask
from pytext.trainers import TrainingState
from pytext.utils import (
    cuda,
    distributed,
    precision,
    round_seq,
    set_random_seeds,
    timing,
)
from pytext.utils.file_io import PathManager


def _set_cuda(
    use_cuda_if_available: bool, device_id: int = 0, world_size: int = 1
) -> None:
    cuda.CUDA_ENABLED = use_cuda_if_available and torch.cuda.is_available()
    cuda.DISTRIBUTED_WORLD_SIZE = world_size

    if use_cuda_if_available and not cuda.CUDA_ENABLED:
        print("Cuda is not available, running on CPU...")
    elif cuda.CUDA_ENABLED:
        torch.cuda.set_device(device_id)

    if device_id == 0:
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


def _set_fp16(use_fp16: bool, rank: int) -> None:
    # only support single GPU training at this moment.
    precision.set_fp16(fp16_enabled=use_fp16)
    if rank == 0:
        print(f"# for debug of FP16: fp16_enabled={precision.FP16_ENABLED}")


def _set_distributed(
    rank: int, world_size: int, dist_init_url: str, device_id: int, gpu_streams: int = 1
) -> None:
    if dist_init_url and world_size > 1:
        distributed.dist_init(
            rank, world_size, dist_init_url, device_id, gpu_streams=gpu_streams
        )


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
    task, training_state = prepare_task(
        config, dist_init_url, device_id, rank, world_size, metric_channels, metadata
    )
    trained_model, best_metric = task.train(config, rank, world_size, training_state)
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
) -> Tuple[Task_Deprecated, TrainingState]:
    if world_size > 1 and config.random_seed is None:
        msg = (
            "Must set random seed when using world_size > 1, so that parameters have "
            "same initialization across workers."
        )
        raise ValueError(msg)

    if rank == 0:
        print("\nParameters: {}\n".format(config), flush=True)
    _set_cuda(config.use_cuda_if_available, device_id, world_size)
    _set_fp16(config.use_fp16, rank)
    _set_distributed(
        rank,
        world_size,
        dist_init_url,
        device_id,
        config.gpu_streams_for_distributed_training,
    )

    if config.random_seed is not None:
        set_random_seeds(config.random_seed, config.use_deterministic_cudnn)

    training_state = None

    if config.auto_resume_from_snapshot:
        # if there are existing checkpoints, resume from the latest one
        latest_snapshot_path = get_latest_checkpoint_path(
            os.path.dirname(config.save_snapshot_path)
        )
        if latest_snapshot_path:
            config.load_snapshot_path = latest_snapshot_path

    if config.load_snapshot_path:
        assert PathManager.isfile(config.load_snapshot_path)
        if config.use_config_from_snapshot:
            task, _, training_state = load(
                config.load_snapshot_path, rank=rank, world_size=world_size
            )
        else:
            task, _, training_state = load(
                config.load_snapshot_path,
                overwrite_config=config,
                rank=rank,
                world_size=world_size,
            )

        if training_state:
            training_state.rank = rank
    else:
        task = create_task(
            config.task, metadata=metadata, rank=rank, world_size=world_size
        )

    for mc in metric_channels or []:
        task.metric_reporter.add_channel(mc)

    return task, training_state


def save_and_export(
    config: PyTextConfig,
    task: Task_Deprecated,
    metric_channels: Optional[List[Channel]] = None,
) -> None:
    print("\n=== Saving model to: " + config.save_snapshot_path)
    meta = None
    tensorizers = None
    if hasattr(task, "data_handler"):
        meta = task.data_handler.metadata_to_save()
    else:
        tensorizers = task.data.tensorizers
    save(config, task.model, meta, tensorizers=tensorizers)
    if len(config.export_list) == 0:
        export_configs = [config.export]
    else:
        export_configs = config.export_list

    for export_config in export_configs:
        if export_config is not None:
            if export_config.export_caffe2_path:
                task.export(
                    task.model,
                    export_config.export_caffe2_path,
                    metric_channels,
                    export_config.export_onnx_path,
                )
            if export_config.export_torchscript_path:
                task.torchscript_export(
                    model=task.model,
                    export_path=export_config.export_torchscript_path,
                    export_config=export_config,
                )
            if export_config.export_lite_path:
                task.lite_export(
                    model=task.model,
                    export_path=export_config.export_lite_path,
                    export_config=export_config,
                )


def export_saved_model_to_caffe2(
    saved_model_path: str, export_caffe2_path: str, output_onnx_path: str = None
) -> None:
    task, train_config, _training_state = load(saved_model_path)
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


def export_saved_model_to_torchscript(
    saved_model_path: str, path: str, export_config: ExportConfig
) -> None:
    task, train_config, _training_state = load(saved_model_path)
    task.torchscript_export(task.model, path, False, 1, export_config=export_config)


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
    task, train_config, _training_state = load(snapshot_path)

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
            test_path, train_config.task.data, field_names, task
        )
        test_results = task.test(data_source)
    else:
        if not test_path:
            test_path = train_config.task.data_handler.test_path
        test_results = task.test(test_path)
    return test_results, test_out_path, metric_channels


def _get_data_source(test_path, data_config, field_names, task):
    if hasattr(data_config, "data_dict_config"):
        # it's multiple data
        if data_config.test_key:
            source_config = data_config.data_dict_config[data_config.test_key].source
        else:
            source_config = next(iter(data_config.data_dict_config.values())).source
    else:
        source_config = getattr(data_config, "source", None)

    if isinstance(task, NewDisjointMultitask):
        # Cannot easily specify a single data source for multitask
        assert not test_path
        data_source = None
    elif test_path and (
        hasattr(source_config, "test_filename") or hasattr(source_config, "test_path")
    ):
        if hasattr(source_config, "test_filename"):
            source_config.test_filename = test_path
        elif hasattr(source_config, "test_path"):
            source_config.test_path = test_path

        if field_names and hasattr(source_config, "field_names"):
            source_config.field_names = field_names
        data_source = create_component(
            ComponentType.DATA_SOURCE, source_config, task.data.data_source.schema
        )
    else:
        data_source = task.data.data_source
    return data_source


class LogitsWriter:
    """Writes model logits to a file.

    The class is designed for use in an asynchronous process spawned by torch.multiprocessing.spawn, e.g.
      logits_writer = LogitsWriter(...)
      logits_writer_ctx = torch.multiprocessing.spawn(logits_writer.run, join=False)
      logits_writer_ctx.join()
    """

    def __init__(
        self,
        results: torch.multiprocessing.Queue,
        output_path: str,
        use_gzip: bool,
        ndigits_precision: int,
    ):
        self.results = results
        self.output_path = output_path
        self.use_gzip = use_gzip
        self.ndigits_precision = ndigits_precision

    def run(self, process_index):
        open_options = self._get_open_options()
        with PathManager.open(self.output_path, **open_options) as fout:
            gzip_fout = (
                gzip.GzipFile(mode="wb", fileobj=fout) if self.use_gzip else None
            )

            while True:
                raw_input_tuple, model_outputs = self.results.get()
                if model_outputs is None:
                    # None means shutdown
                    break

                if self.ndigits_precision:
                    model_outputs = round_seq(model_outputs, self.ndigits_precision)

                # multi-encoder output
                if isinstance(model_outputs, tuple):
                    self._write(fout, gzip_fout, zip(*raw_input_tuple, *model_outputs))
                # single encoder output
                elif isinstance(model_outputs, list):
                    self._write(fout, gzip_fout, zip(*raw_input_tuple, model_outputs))
                else:
                    raise Exception("Expecting tuple or tensor types for model_outputs")

            if self.use_gzip:
                gzip_fout.close()

    def _get_open_options(self):
        """We must open the file in binary model for gzip"""
        if self.use_gzip:
            return {"mode": "wb"}
        else:
            return {"mode": "w", "encoding": "utf-8"}

    def _write(
        self,
        fout: Union[IO[str], IO[bytes]],
        gzip_fout: gzip.GzipFile,
        rows: Iterator[Any],
    ):
        """Conditionally write to gzip or normal text file depending on the settings."""
        for row in rows:
            dump_row = "\t".join(json.dumps(r) for r in row)
            text = f"{dump_row}\n"

            if self.use_gzip:
                gzip_fout.write(text.encode())
            else:
                fout.write(text)


def get_logits(
    snapshot_path: str,
    use_cuda_if_available: bool,
    output_path: Optional[str] = None,
    test_path: Optional[str] = None,
    field_names: Optional[List[str]] = None,
    dump_raw_input: bool = False,
    batch_size: int = 16,
    ndigits_precision: int = 0,
    output_columns: Optional[List[int]] = None,
    use_gzip: bool = False,
    device_id: int = 0,
    fp16: bool = False,
):
    _set_cuda(use_cuda_if_available, device_id)
    task, train_config, _training_state = load(snapshot_path)
    print(f"Successfully loaded model from {snapshot_path}")
    print(f"Model on GPU? {next(task.model.parameters()).is_cuda}")
    print(f"CUDA device id: {torch.cuda.current_device()}")

    if isinstance(task, NewTask):
        task.model.eval()

        if fp16:
            task.model.half()

        data_source = _get_data_source(
            test_path, train_config.task.data, field_names, task
        )
        task.data.batcher = Batcher(test_batch_size=batch_size)
        task.data.sort_key = None
        batches = task.data.batches(Stage.TEST, data_source=data_source)

        mp = torch.multiprocessing.get_context("spawn")

        with torch.no_grad():
            results = mp.SimpleQueue()
            logits_writer = LogitsWriter(
                results, output_path, use_gzip, ndigits_precision
            )
            logits_writer_ctx = torch.multiprocessing.spawn(
                logits_writer.run, join=False
            )

            for (raw_batch, tensor_dict) in batches:
                raw_input_tuple = (
                    dict_zip(*raw_batch, value_only=True) if dump_raw_input else ()
                )

                model_inputs = task.model.arrange_model_inputs(tensor_dict)
                model_outputs = task.model(*model_inputs)

                # multi-encoder output
                if isinstance(model_outputs, tuple):
                    # prevent breaking behaviour in default case
                    output_columns = (
                        range(len(model_outputs))
                        if not output_columns
                        else output_columns
                    )
                    model_outputs = tuple(
                        m.tolist()
                        for i, m in enumerate(model_outputs)
                        if i in output_columns
                    )
                # single encoder output
                elif isinstance(model_outputs, list):
                    model_outputs = model_outputs.tolist()
                else:
                    raise Exception("Expecting tuple or tensor types for model_outputs")

                results.put((raw_input_tuple, model_outputs))

            results.put((None, None))
            logits_writer_ctx.join()
            print(
                f"Finished logits generation for file {test_path} with output {output_path}"
            )


def save_pytext_snapshot(config: PyTextConfig) -> None:
    task, training_state = prepare_task(
        config,
        dist_init_url=None,
        device_id=0,
        rank=0,
        world_size=1,
        metric_channels=None,
        metadata=None,
    )
    print("Task set up successful.\n")
    save_and_export(config, task)


def dict_zip(*dicts, value_only=False):
    dict_keys = dicts[0].keys()
    return (
        tuple([d[k] for d in dicts] for k in dict_keys)
        if value_only
        else {k: [d[k] for d in dicts] for k in dict_keys}
    )


def batch_predict(model_file: str, examples: List[Dict[str, Any]]):
    task, train_config, _training_state = load(model_file)
    return task.predict(examples)

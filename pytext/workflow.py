#!/usr/bin/env python3
import os
from typing import Any, List, Optional, Tuple, get_type_hints

import torch
from pytext.config import PyTextConfig, TestConfig
from pytext.config.component import (
    create_data_handler,
    create_exporter,
    create_featurizer,
    create_metric_reporter,
    create_model,
    create_trainer,
)
from pytext.config.field_config import LabelConfig
from pytext.data import BatchIterator, DataHandler
from pytext.exporters.exporter import ModelExporter
from pytext.metric_reporters import MetricReporter
from pytext.models.embeddings.token_embedding import FeatureConfig
from pytext.optimizer import create_optimizer
from pytext.optimizer.scheduler import Scheduler
from pytext.trainers.trainer import Trainer
from pytext.utils.dist_utils import dist_init

from .serialize import load, save
from .utils import cuda_utils


class Job:
    __slots__ = [
        "trainer",
        "data_handler",
        "train_iter",
        "eval_iter",
        "model",
        "metric_reporter",
        "optimizers",
        "metadata",
        "lr_scheduler",
    ]

    def __init__(
        self,
        trainer: Trainer,
        data_handler: DataHandler,
        train_iter: BatchIterator,
        eval_iter: BatchIterator,
        model: torch.nn.Module,
        metric_reporter: MetricReporter,
        optimizers: List[torch.optim.Optimizer],
        lr_scheduler: Scheduler,
    ) -> None:
        self.trainer: Trainer = trainer
        self.data_handler: DataHandler = data_handler
        self.train_iter: BatchIterator = train_iter
        self.eval_iter: BatchIterator = eval_iter
        self.model: torch.nn.Module = model
        self.metric_reporter: MetricReporter = metric_reporter
        self.optimizers: List[torch.optim.Optimizer] = optimizers
        self.lr_scheduler: Scheduler = lr_scheduler


def _set_cuda(
    use_cuda_if_available: bool, device_id: int = 0, world_size: int = 1
) -> None:
    cuda_utils.CUDA_ENABLED = use_cuda_if_available and torch.cuda.is_available()
    cuda_utils.DISTRIBUTED_WORLD_SIZE = world_size

    if use_cuda_if_available and not cuda_utils.CUDA_ENABLED:
        print("Cuda is not available, running on CPU...")
    elif cuda_utils.CUDA_ENABLED:
        torch.cuda.set_device(device_id)

    # for debug of GPU
    print(
        """
    use_cuda_if_available: {}\n
    device_id: {}\n
    world_size: {}\n
    torch.cuda.is_available(): {}\n
    cuda_utils.CUDA_ENABLED: {}\n
    cuda_utils.DISTRIBUTED_WORLD_SIZE: {}\n
    """.format(
            use_cuda_if_available,
            device_id,
            world_size,
            torch.cuda.is_available(),
            cuda_utils.CUDA_ENABLED,
            cuda_utils.DISTRIBUTED_WORLD_SIZE,
        )
    )


def train_model(
    config: PyTextConfig,
    dist_init_url: str = None,
    device_id: int = 0,
    rank: int = 0,
    world_size: int = 1,
) -> Tuple:
    job = prepare_job(config, dist_init_url, device_id, rank, world_size)
    trained_model, best_metric = job.trainer.train(
        job.train_iter,
        job.eval_iter,
        job.model,
        job.metric_reporter,
        job.optimizers,
        config,
        job.lr_scheduler,
    )
    # Only rank 0 gets to finalize the job and export the model
    if rank == 0:
        finalize_job(config, trained_model, job.data_handler)
    return trained_model, best_metric


def prepare_job(
    config: PyTextConfig,
    dist_init_url: str = None,
    device_id: int = 0,
    rank: int = 0,
    world_size: int = 1,
) -> Job:

    if dist_init_url:
        dist_init(rank, world_size, dist_init_url)

    print("\nParameters:\n{}".format(config))
    _set_cuda(config.use_cuda_if_available, device_id, world_size)
    jobspec = config.jobspec
    featurizer = create_featurizer(jobspec.featurizer, jobspec.features)
    # load data
    data_handler = create_data_handler(
        jobspec.data_handler, jobspec.features, jobspec.labels, featurizer=featurizer
    )

    print("\nLoading data...")
    data_handler.init_metadata()

    train_iter = data_handler.get_train_iter(rank, world_size)
    eval_iter = data_handler.get_eval_iter()

    # load or create model
    metadata = data_handler.metadata
    if config.load_snapshot_path is None or not os.path.isfile(
        config.load_snapshot_path
    ):
        model = create_model(jobspec.model, jobspec.features, metadata)
    else:
        print("\nLoading model from [%s]..." % config.load_snapshot_path)
        model = load(config.load_snapshot_path)["model"]

    optimizers = create_optimizer(model, jobspec.optimizer)
    metric_reporter = create_metric_reporter(config.jobspec.metric_reporter, metadata)
    lr_scheduler = Scheduler(
        optimizers, jobspec.scheduler, metric_reporter.lower_is_better
    )
    trainer = create_trainer(jobspec.trainer)
    return Job(
        trainer,
        data_handler,
        train_iter,
        eval_iter,
        model,
        metric_reporter,
        optimizers,
        lr_scheduler,
    )


def finalize_job(
    config: PyTextConfig, trained_model: torch.nn.Module, data_handler: DataHandler
) -> None:
    print("Saving pytorch model to: " + config.save_snapshot_path)
    # Make sure to put the model on CPU and disable CUDA before exporting to
    # ONNX to disable any data_parallel pieces
    _set_cuda(False)
    trained_model = trained_model.cpu()
    save(config.save_snapshot_path, config, trained_model, data_handler)

    if config.jobspec.exporter:
        export_to_caffe2(
            config.jobspec.exporter,
            config.jobspec.features,
            config.jobspec.labels,
            trained_model,
            data_handler,
            config.export_caffe2_path,
        )


def export_saved_model_to_caffe2(
    saved_model_path: str, export_caffe2_path: str
) -> None:
    config, model, data_handler = load(saved_model_path)
    exporter = config.jobspec.exporter
    if exporter is None:
        JobspecType = type(config.jobspec)
        ExporterConfigType = get_type_hints(JobspecType)["exporter"].__args__[0]
        exporter = ExporterConfigType()

    export_to_caffe2(
        exporter,
        config.jobspec.features,
        config.jobspec.labels,
        model,
        data_handler,
        export_caffe2_path,
    )


def export_to_caffe2(
    exporter: ModelExporter,
    features: FeatureConfig,
    labels: Optional[LabelConfig],
    trained_model: torch.nn.Module,
    data_handler: DataHandler,
    export_caffe2_path: str,
) -> None:
    print("Saving caffe2 model to: " + export_caffe2_path)
    exporter = create_exporter(exporter, features, labels, data_handler.metadata)
    exporter.export_to_caffe2(trained_model, export_caffe2_path)


def test_model(test_config: TestConfig) -> Any:
    _set_cuda(test_config.use_cuda_if_available)
    model_path = test_config.load_snapshot_path
    if model_path is None or not os.path.isfile(model_path):
        raise ValueError("Invalid snapshot path for testing: {}".format(model_path))

    print("\nLoading model from [%s]..." % model_path)

    train_config, model, data_handler = load(model_path)
    model.eval()

    if cuda_utils.CUDA_ENABLED:
        model = model.cuda()
    trainer = create_trainer(train_config.jobspec.trainer)

    # TODO can set different channel for test
    metric_reporter = create_metric_reporter(
        train_config.jobspec.metric_reporter, data_handler.metadata
    )
    data_handler.test_path = test_config.test_path
    test_iter = data_handler.get_test_iter()
    return (
        trainer.test(test_iter, model, metric_reporter),
        train_config.jobspec.metric_reporter.output_path,
    )

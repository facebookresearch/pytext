#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Dict, Optional

from pytext.common.constants import Stage
from pytext.config import ConfigBase, PyTextConfig
from pytext.config.component import ComponentType, create_component
from pytext.data import types as data_types
from pytext.data.data import Data
from pytext.data.sources import DataSchema
from pytext.data.tensorizers import Tensorizer
from pytext.exporters import ModelExporter
from pytext.metric_reporters import ClassificationMetricReporter, MetricReporter
from pytext.models.doc_model import NewDocModel as DocModel
from pytext.models.model import BaseModel as Model
from pytext.optimizer import Adam, Optimizer
from pytext.optimizer.scheduler import Scheduler
from pytext.trainers import Trainer
from pytext.utils import cuda_utils, time_utils

from .task import TaskBase


class NewTaskTrainer(Trainer):
    class Config(Trainer.Config):
        """Make mypy happy"""

    def _run_epoch(
        self,
        stage: Stage,
        epoch: int,
        batches,
        model: Model,
        metric_reporter: MetricReporter,
        pre_batch=lambda: None,
        backprop=lambda loss: None,
        rank=0,
        num_samples_to_log_progress: int = None,
    ):
        """Our run_epoch is a bit different, because we're wrapping the model forward
        call with model.train_batch, which arranges tensors and gets loss, etc."""
        print(f"Rank {rank} worker: Running epoch #{epoch} for {stage}")
        report_metric = stage != Stage.TRAIN or self.config.report_train_metrics

        for batch_id, batch in enumerate(batches):
            pre_batch()
            with time_utils.time("model.train_batch"):
                loss, metric_data = model.train_batch(batch)
            with time_utils.time("backprop"):
                backprop(loss)
            if report_metric:
                with time_utils.time("add metrics"):
                    metric_reporter.add_batch_stats(
                        batch_id, *metric_data, **metric_reporter.batch_context(batch)
                    )

        metrics = None
        if report_metric:
            with time_utils.time("report metrics"):
                metrics = metric_reporter.report_metric(
                    model, stage, epoch, print_to_channels=(rank == 0)
                )
        else:
            metric_reporter._reset()

        return metrics

    def _prepare_scheduler(self, training_batches, scheduler=None):
        """Batch based schedulers require knowing the number of batches in
        the data. We're not supporting that yet with the Data api, need to figure out
        how to expose this info or restructure batch-based schedulers to not need it."""
        if scheduler.batch_based_schedulers:
            raise Exception("New tasks don't yet support batch-based scheduling")
        return scheduler


class NewTask(TaskBase):
    """This task abstraction separates the concerns into three main components,
    `pytext.data.Data`, `pytext.models.new_model.NewModel` (names and paths
    will change as these become the primary abstractions), and `pytext.trainers.Trainer`

    At its simplest, this abstraction is as follows:
    - `Task` defines a `DataSchema` which describes the python types which are
    accessible in each row of the input data. This should be a common data schema
    which describes a single NLP task, each of which has the same or sufficiently
    similar inputs/outputs (think document classification, optionally including
    dense features).
    - `Model` defines an input signature (called `tensorizers`) which describes the
    tensors that it wants to execute each batch for training or prediction.
    - `Data` takes the `DataSchema` and `tensorizers` and exposes `train`, `test`, and
    `eval` attributes. Iterating over these attributes yields a batch, which is a
    dictionary with the same keys as the `tensorizers` input signature, and whose
    values are tensors.
    - `Model` can train or predict using these input batch dictionaries directly,
    and interally is responsible for defining how to arrange the tensors for passing
    to its forward functions and loss generation. Its train_batch function returns
    a tuple of `(loss, metrics)`, where `loss` is a `torch.loss.Loss` instance, and
    `metrics` can be passed directly into a compatible metric reporter.
    - `Trainer` contains the core training logic, iterating through batches created by
    `Data`, passing them to `Model` for training, aggregating and reporting metrics,
    running `loss.backward()` and `optimizer.step()`, and managing the scheduler.
    """

    __EXPANSIBLE__ = True

    DATA_SCHEMA: DataSchema

    class Config(ConfigBase):
        data: Data.Config = Data.Config()
        model: Model.Config
        trainer: NewTaskTrainer.Config = NewTaskTrainer.Config()
        optimizer: Optimizer.Config = Adam.Config()
        scheduler: Scheduler.Config = Scheduler.Config()
        exporter: Optional[ModelExporter.Config] = None

    @classmethod
    def from_config(cls, config: Config, unused_metadata=None, model_state=None):
        tensorizers = {
            name: create_component(ComponentType.TENSORIZER, tensorizer)
            for name, tensorizer in config.model.inputs._asdict().items()
        }
        # This initializes the tensorizers
        data = create_component(
            ComponentType.DATA_HANDLER, config.data, cls.DATA_SCHEMA, tensorizers
        )
        # Initialized tensorizers can be used to create the model
        model = create_component(ComponentType.MODEL, config.model, tensorizers)
        if model_state:
            model.load_state_dict(model_state)
        if cuda_utils.CUDA_ENABLED:
            model = model.cuda()
        # This is the only place right now that the task actually cares about which
        # features and tensors are being used. This is a strong tie between
        # the implementation of the model and the metric reporter.
        metric_reporter = cls.create_metric_reporter(config, tensorizers)
        trainer = create_component(ComponentType.TRAINER, config.trainer)
        optimizer = create_component(ComponentType.OPTIMIZER, config.optimizer, model)
        scheduler = Scheduler(
            optimizer, config.scheduler, metric_reporter.lower_is_better
        )
        if config.exporter:
            exporter = create_component(ComponentType.EXPORTER, config.exporter)
        else:
            exporter = None
        return cls(
            data, model, metric_reporter, trainer, optimizer, scheduler, exporter
        )

    def __init__(
        self,
        data: Data,
        model: Model,
        metric_reporter: Optional[MetricReporter] = None,
        trainer: Optional[NewTaskTrainer] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[Scheduler] = None,
        exporter: Optional[ModelExporter] = None,
    ):
        self.data = data
        self.model = model
        # Attempt to build a default metric reporter
        self.metric_reporter = metric_reporter or self.create_metric_reporter(
            self.Config.metric_reporter, model
        )
        self.trainer = trainer or NewTaskTrainer()
        self.optimizer = optimizer or Adam(
            model.parameters(), **Adam.Config()._asdict()
        )
        self.scheduler = scheduler
        self.exporter = exporter

    def train(self, config: PyTextConfig, rank: int = 0, unused_world_size: int = 1):
        return self.trainer.train(
            self.data.batches(Stage.TRAIN),
            self.data.batches(Stage.EVAL),
            self.model,
            self.metric_reporter,
            config,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            rank=rank,
        )

    def test(self):
        return self.trainer.test(
            self.data.batches(Stage.TEST), self.model, self.metric_reporter
        )


class NewDocumentClassification(NewTask):
    DATA_SCHEMA: DataSchema = {"text": data_types.Text, "label": data_types.Label}

    class Config(NewTask.Config):
        model: Model.Config = DocModel.Config()
        metric_reporter: ClassificationMetricReporter.Config = (
            ClassificationMetricReporter.Config()
        )

    # The existence of this function is a pretty good argument for having
    # the metric reporter be owned internally at least in some way by the model
    @classmethod
    def create_metric_reporter(cls, config: Config, tensorizers: Dict[str, Tensorizer]):
        return ClassificationMetricReporter.from_config_and_label_names(
            config.metric_reporter, list(tensorizers["labels"].labels)
        )

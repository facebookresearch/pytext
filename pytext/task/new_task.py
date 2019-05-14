#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Any, Dict, List, Optional, Type

from pytext.common.constants import Stage
from pytext.config import ConfigBase, PyTextConfig
from pytext.config.component import ComponentType, create_component, create_trainer
from pytext.data.data import Data
from pytext.metric_reporters import (
    ClassificationMetricReporter,
    MetricReporter,
    RegressionMetricReporter,
)
from pytext.models.doc_model import (
    NewDocModel as DocModel,
    NewDocRegressionModel as DocRegressionModel,
)
from pytext.models.model import BaseModel as Model
from pytext.models.pair_classification_model import BasePairwiseModel, PairwiseModel
from pytext.trainers import Trainer, TrainingState
from pytext.utils import cuda, precision, timing
from torch import jit

from .task import TaskBase


class NewTaskTrainer(Trainer):
    class Config(Trainer.Config):
        """Make mypy happy"""

    @timing.time("run_step")
    def run_step(
        self,
        samples: List[Any],
        state: TrainingState,
        metric_reporter: MetricReporter,
        report_metric: bool,
    ):
        """Our run_step is a bit different, because we're wrapping the model forward
        call with model.train_batch, which arranges tensors and gets loss, etc."""
        sample_size = len(samples)
        assert sample_size <= self.config.num_accumulated_batches

        model = state.model
        self.zero_grads(state)
        metrics_inputs = []
        for i, (batch_id, batch) in enumerate(samples):
            if cuda.DISTRIBUTED_WORLD_SIZE > 1:
                # Whenever *samples* contains more than one mini-batch, we
                # want to accumulate gradients locally and only call
                # all-reduce in the last backwards pass.
                if i < sample_size - 1:
                    # sync gradients in the last sample backward
                    model.accumulate_gradients(True)
                else:
                    model.accumulate_gradients(False)

            with timing.time("model.train_batch"):
                loss, metric_data = model.train_batch(model, batch)
                if sample_size > 1:
                    # gradients averaged per each batch and accumulated across samples.
                    # divide sample_size to let gradients averaged per example
                    loss = loss / sample_size
            self.backprop(state, loss)

            if report_metric:
                # don't add metrics yet, in case there is oveflow
                metrics_inputs.append((batch_id, metric_data, batch))

        # update gradients after #len(samples) forward & backward
        self.optimizer_step(state)

        # add metrics if batch is not skipped due to overflow
        with timing.time("add metrics"):
            if state.stage != Stage.TRAIN or not getattr(
                state.optimizer, "overflow", False
            ):
                for metrics_input in metrics_inputs:
                    batch_id, metric_data, batch = metrics_input
                    metric_reporter.add_batch_stats(
                        batch_id, *metric_data, **metric_reporter.batch_context(batch)
                    )
            metrics_inputs = []

    def _prepare_scheduler(self, training_batches, scheduler=None):
        """Batch based schedulers require knowing the number of batches in
        the data. We're not supporting that yet with the Data api, need to figure out
        how to expose this info or restructure batch-based schedulers to not need it."""
        if scheduler.batch_based_schedulers:
            raise Exception("New tasks don't yet support batch-based scheduling")
        return scheduler


class _NewTask(TaskBase):
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

    class Config(ConfigBase):
        data: Data.Config = Data.Config()
        trainer: NewTaskTrainer.Config = NewTaskTrainer.Config()

    @classmethod
    def from_config(
        cls,
        config: Config,
        unused_metadata=None,
        model_state=None,
        rank=0,
        world_size=1,
    ):
        tensorizers, data = NewTask._init_tensorizers(config, rank, world_size)

        # Initialized tensorizers can be used to create the model
        model = NewTask._init_model(config, tensorizers, model_state)

        # This is the only place right now that the task actually cares about which
        # features and tensors are being used. This is a strong tie between
        # the implementation of the model and the metric reporter.
        metric_reporter = create_component(
            ComponentType.METRIC_REPORTER,
            config.metric_reporter,
            tensorizers=tensorizers,
        )
        trainer = create_trainer(config.trainer, model)
        return cls(data, model, metric_reporter, trainer)

    @classmethod
    def _init_tensorizers(cls, config: Config, rank, world_size):
        tensorizers = {
            name: create_component(ComponentType.TENSORIZER, tensorizer_config)
            for name, tensorizer_config in config.model.inputs._asdict().items()
            if tensorizer_config
        }
        schema: Dict[str, Type] = {}
        for tensorizer in tensorizers.values():
            for name, type in tensorizer.column_schema:
                if name in schema and type != schema[name]:
                    raise TypeError(f"Expected two different types for column {name}")
                schema[name] = type

        # This initializes the tensorizers
        data = create_component(
            ComponentType.DATA_HANDLER,
            config.data,
            schema,
            tensorizers,
            rank=rank,
            world_size=world_size,
        )
        return tensorizers, data

    @classmethod
    def _init_model(cls, config: Config, tensorizers, model_state=None):
        model = create_component(
            ComponentType.MODEL, config.model, tensorizers=tensorizers
        )
        if model_state:
            model.load_state_dict(model_state)

        precision.activate(model)
        if cuda.CUDA_ENABLED:
            model = model.cuda()

        return model

    def __init__(
        self,
        data: Data,
        model: Model,
        metric_reporter: Optional[MetricReporter] = None,
        trainer: Optional[NewTaskTrainer] = None,
    ):
        self.data = data
        self.model = model
        # Attempt to build a default metric reporter
        self.metric_reporter = metric_reporter or self.create_metric_reporter(
            self.Config.metric_reporter, model
        )
        self.trainer = trainer or NewTaskTrainer()

    def train(self, config: PyTextConfig, rank: int = 0, world_size: int = 1):
        # TODO: move dist_init back to prepare_task in pytext/workflow.py
        # when processing time between dist_init and first loss.backward() is short
        return self.trainer.train(
            self.data.batches(Stage.TRAIN),
            self.data.batches(Stage.EVAL),
            self.model,
            self.metric_reporter,
            config,
            rank=rank,
        )

    def test(self, data_source):
        return self.trainer.test(
            self.data.batches(Stage.TEST, data_source=data_source),
            self.model,
            self.metric_reporter,
        )

    def export(self, model, export_path, metric_channels=None, export_onnx_path=None):
        # Make sure to put the model on CPU and disable CUDA before exporting to
        # ONNX to disable any data_parallel pieces
        cuda.CUDA_ENABLED = False
        model = model.cpu()
        precision.deactivate(model)

        batch = next(iter(self.data.batches(Stage.TRAIN)))
        print(f"Saving caffe2 model to: {export_path}")
        return model.caffe2_export(
            self.data.tensorizers, batch, export_path, export_onnx_path=export_onnx_path
        )

    def torchscript_export(self, model, export_path):
        # Make sure to put the model on CPU and disable CUDA before exporting to
        # ONNX to disable any data_parallel pieces
        cuda.CUDA_ENABLED = False
        model.cpu()
        precision.deactivate(model)
        # Trace needs eval mode, to disable dropout etc
        model.eval()
        model.prepare_for_onnx_export_()

        batch = next(iter(self.data.batches(Stage.TEST)))
        inputs = model.arrange_model_inputs(batch)
        trace = jit.trace(model, inputs)
        if hasattr(model, "torchscriptify"):
            trace = model.torchscriptify(self.data.tensorizers, trace)
        print(f"Saving torchscript model to: {export_path}")
        trace.save(export_path)


class NewTask(_NewTask):
    __EXPANSIBLE__ = True

    class Config(_NewTask.Config):
        model: Model.Config


class NewDocumentClassification(NewTask):
    class Config(NewTask.Config):
        model: Model.Config = DocModel.Config()
        metric_reporter: ClassificationMetricReporter.Config = (
            ClassificationMetricReporter.Config()
        )


class NewDocumentRegression(NewTask):
    class Config(NewTask.Config):
        model: Model.Config = DocRegressionModel.Config()
        metric_reporter: RegressionMetricReporter.Config = (
            RegressionMetricReporter.Config()
        )


class PairwiseClassification(NewTask):
    class Config(NewTask.Config):
        model: BasePairwiseModel.Config = PairwiseModel.Config()
        metric_reporter: ClassificationMetricReporter.Config = (
            ClassificationMetricReporter.Config()
        )

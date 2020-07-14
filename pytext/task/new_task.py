#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Any, Dict, Optional, Type, Union

import torch
from pytext.common.constants import Stage
from pytext.config import ConfigBase, PyTextConfig
from pytext.config.component import ComponentType, create_component, create_trainer
from pytext.data.data import Data
from pytext.data.sources.data_source import Schema
from pytext.data.tensorizers import Tensorizer
from pytext.metric_reporters import MetricReporter
from pytext.models.model import BaseModel
from pytext.trainers import TaskTrainer, TrainingState
from pytext.utils import cuda, onnx
from pytext.utils.usage import log_class_usage
from torch import jit, sort

from .task import TaskBase


def create_schema(
    tensorizers: Dict[str, Tensorizer], extra_schema: Optional[Dict[str, Type]] = None
) -> Schema:
    schema: Dict[str, Type] = {}

    def add_to_schema(name, type):
        if name in schema:
            if type != Any and type != schema[name]:
                raise TypeError(
                    f"Unexpected different types for column {name}: "
                    f"{type} != {schema[name]}"
                )
        else:
            schema[name] = type

    for tensorizer in tensorizers.values():
        for name, type in tensorizer.column_schema:
            add_to_schema(name, type)

    for name, type in (extra_schema or {}).items():
        # Schema type check is not needed, ClassificationMetricReporter
        # automatically casts data to string.
        add_to_schema(name, type)

    print(f"PyText data schema: {schema}.")
    return schema


def create_tensorizers(
    model_inputs: Union[BaseModel.Config.ModelInput, Dict[str, Tensorizer.Config]]
) -> Dict[str, Tensorizer]:
    if not isinstance(model_inputs, dict):
        model_inputs = model_inputs._asdict()

    tensorizers = {
        name: create_component(ComponentType.TENSORIZER, tensorizer_config)
        for name, tensorizer_config in model_inputs.items()
        if tensorizer_config
    }

    return tensorizers


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
        trainer: TaskTrainer.Config = TaskTrainer.Config()
        # TODO: deprecate this
        use_elastic: Optional[bool] = None

    @classmethod
    def from_config(
        cls,
        config: Config,
        unused_metadata=None,
        model_state=None,
        tensorizers=None,
        rank=0,
        world_size=1,
    ):
        print(f"Creating task: {cls.__name__}...")
        tensorizers, data = cls._init_tensorizers(config, tensorizers, rank, world_size)

        # Initialized tensorizers can be used to create the model
        model = cls._init_model(config.model, tensorizers, model_state)

        # This is the only place right now that the task actually cares about which
        # features and tensors are being used. This is a strong tie between
        # the implementation of the model and the metric reporter.
        metric_reporter = cls.create_metric_reporter(config, tensorizers)
        trainer = create_trainer(config.trainer, model)
        return cls(data, model, metric_reporter, trainer)

    @classmethod
    def create_metric_reporter(cls, config, tensorizers):
        return create_component(
            ComponentType.METRIC_REPORTER,
            config.metric_reporter,
            tensorizers=tensorizers,
        )

    @classmethod
    def _init_tensorizers(cls, config: Config, tensorizers=None, rank=0, world_size=1):
        # Pull extra columns from the metric reporter config to pass into
        # the data source schema.
        extra_columns = (
            getattr(config.metric_reporter, "text_column_names", [])
            + getattr(config.metric_reporter, "additional_column_names", [])
            + getattr(config.metric_reporter, "student_column_names", [])
        )
        extra_schema = {column: Any for column in extra_columns}

        init_tensorizers = not tensorizers
        if init_tensorizers:
            tensorizers = create_tensorizers(config.model.inputs)

        schema = create_schema(tensorizers, extra_schema)

        # This initializes the tensorizers
        data = create_component(
            ComponentType.DATA_HANDLER,
            config.data,
            schema,
            tensorizers,
            rank=rank,
            world_size=world_size,
            init_tensorizers=init_tensorizers,
        )
        return tensorizers, data

    @classmethod
    def _init_model(cls, model_config, tensorizers, model_state=None):
        model_config.init_from_saved_state = model_state is not None
        model = create_component(
            ComponentType.MODEL, model_config, tensorizers=tensorizers
        )
        if model_state:
            print("Loading model from model state dict...")
            model.load_state_dict(model_state)
            print("Loaded!")

        if cuda.CUDA_ENABLED:
            model = model.cuda()

        return model

    def __init__(
        self,
        data: Data,
        model: BaseModel,
        metric_reporter: Optional[MetricReporter] = None,
        trainer: Optional[TaskTrainer] = None,
    ):
        self.data = data
        self.model = model
        # Attempt to build a default metric reporter
        self.metric_reporter = metric_reporter or self.create_metric_reporter(
            self.Config.metric_reporter, model
        )
        self.trainer = trainer or TaskTrainer()
        log_class_usage

    def train(
        self,
        config: PyTextConfig,
        rank: int = 0,
        world_size: int = 1,
        training_state: TrainingState = None,
    ):
        # next to move dist_init back to prepare_task in pytext/workflow.py
        # when processing time between dist_init and first loss.backward() is short
        if training_state:
            return self.trainer.train_from_state(
                training_state,
                self.data.batches(Stage.TRAIN),
                self.data.batches(Stage.EVAL),
                self.metric_reporter,
                config,
            )
        return self.trainer.train(
            self.data.batches(Stage.TRAIN),
            self.data.batches(Stage.EVAL),
            self.model,
            self.metric_reporter,
            config,
            rank=rank,
        )

    @torch.no_grad()
    def test(self, data_source):
        return self.trainer.test(
            self.data.batches(Stage.TEST, data_source=data_source),
            self.model,
            self.metric_reporter,
        )

    def predict(self, examples):
        """
        Generates predictions using PyTorch model. The difference with `test()` is
        that this should be used when the the examples do not have any true
        label/target.

        Args:
            examples: json format examples, input names should match the names specified
                in this task's features config
        """
        self.model.eval()
        results = []
        input_tensorizers = {
            name: tensorizer
            for name, tensorizer in self.data.tensorizers.items()
            if tensorizer.is_input
        }
        for row in examples:
            numberized_row = {
                name: [tensorizer.numberize(row)]
                for name, tensorizer in input_tensorizers.items()
            }
            tensor_dict = {
                name: tensorizer.tensorize(batch=numberized_row[name])
                for name, tensorizer in input_tensorizers.items()
            }
            model_inputs = self.model.arrange_model_inputs(tensor_dict)
            model_context = self.model.arrange_model_context(tensor_dict)
            predictions, scores = self.model.get_pred(
                self.model(*model_inputs), context=model_context
            )
            results.append({"prediction": predictions, "score": scores})
        return results

    def export(self, model, export_path, metric_channels=None, export_onnx_path=None):
        # Make sure to put the model on CPU and disable CUDA before exporting to
        # ONNX to disable any data_parallel pieces
        onnx.validate_onnx_export(model)

        cuda.CUDA_ENABLED = False
        model = model.cpu()
        optimizer = self.trainer.optimizer
        optimizer.pre_export(model)

        unused_raw_batch, batch = next(
            iter(self.data.batches(Stage.TRAIN, load_early=True))
        )
        if metric_channels:
            print("Exporting metrics")
            for mc in metric_channels:
                mc.export(model, model.arrange_model_inputs(batch))

        return model.caffe2_export(
            self.data.tensorizers, batch, export_path, export_onnx_path=export_onnx_path
        )

    def torchscript_export(
        self, model, export_path=None, quantize=False, sort_input=False, sort_key=1
    ):
        # Make sure to put the model on CPU and disable CUDA before exporting to
        # ONNX to disable any data_parallel pieces
        cuda.CUDA_ENABLED = False
        model.cpu()
        optimizer = self.trainer.optimizer
        optimizer.pre_export(model)

        # Trace needs eval mode, to disable dropout etc
        model.eval()
        model.prepare_for_onnx_export_()

        unused_raw_batch, batch = next(
            iter(self.data.batches(Stage.TRAIN, load_early=True))
        )
        inputs = model.onnx_trace_input(batch)
        # call model forward to set correct device types
        if sort_input:
            _, sorted_indices = sort(inputs[sort_key], descending=True)
            inputs = [i.index_select(0, sorted_indices) for i in inputs]
        model(*inputs)
        if quantize:
            model.quantize()
        trace = jit.trace(model, inputs)
        if hasattr(model, "torchscriptify"):
            trace = model.torchscriptify(self.data.tensorizers, trace)
        trace.apply(lambda s: s._pack() if s._c._has_method("_pack") else None)
        if export_path is not None:
            print(f"Saving torchscript model to: {export_path}")
            trace.save(export_path)
        return trace


class NewTask(_NewTask):
    __EXPANSIBLE__ = True

    class Config(_NewTask.Config):
        model: BaseModel.Config

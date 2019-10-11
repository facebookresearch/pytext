#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import List, Optional

from pytext.common.constants import BatchContext
from pytext.config import ConfigBase
from pytext.config.component import (
    Component,
    ComponentType,
    create_component,
    create_data_handler,
    create_exporter,
    create_featurizer,
    create_metric_reporter,
    create_model,
    create_trainer,
)
from pytext.config.field_config import FeatureConfig
from pytext.data import DataHandler
from pytext.data.featurizer import Featurizer, SimpleFeaturizer
from pytext.exporters import ModelExporter
from pytext.loss import KLDivergenceBCELoss, KLDivergenceCELoss
from pytext.metric_reporters import MetricReporter
from pytext.models import Model
from pytext.trainers import Trainer
from pytext.utils import cuda, lazy


def create_task(
    task_config, metadata=None, model_state=None, tensorizers=None, rank=0, world_size=1
):
    """
    Create a task by finding task class in registry and invoking the from_config
    function of the class, see :meth:`~Task.from_config` for more details
    """
    return create_component(
        ComponentType.TASK,
        task_config,
        metadata,
        model_state,
        tensorizers=tensorizers,
        rank=rank,
        world_size=world_size,
    )


class TaskBase(Component):
    """
    Task is the central place to define and wire up components for data processing,
    model training, metric reporting, etc. Task class has a Config class containing
    the config of each component in a descriptive way.
    """

    __COMPONENT_TYPE__ = ComponentType.TASK

    class Config(ConfigBase):
        features: FeatureConfig = FeatureConfig()
        featurizer: Featurizer.Config = SimpleFeaturizer.Config()
        data_handler: DataHandler.Config
        trainer: Trainer.Config = Trainer.Config()
        exporter: Optional[ModelExporter.Config] = None

    @classmethod
    def from_config(
        cls,
        task_config,
        metadata=None,
        model_state=None,
        tensorizers=None,
        rank=1,
        world_size=0,
    ):
        """
        Create the task from config, and optionally load metadata/model_state
        This function will create components including :class:`~DataHandler`,
        :class:`~Trainer`, :class:`~MetricReporter`,
        :class:`~Exporter`, and wire them up.

        Args:
            task_config (Task.Config): the config of the current task
            metadata: saved global context of this task, e.g: vocabulary, will be
                generated by :class:`~DataHandler` if it's None
            model_state: saved model parameters, will be loaded into model when given
        """
        if hasattr(task_config.labels, "target_prob"):
            assert task_config.labels.target_prob == isinstance(
                task_config.model.output_layer.loss,
                (KLDivergenceBCELoss.Config, KLDivergenceCELoss.Config),
            ), "target_prob must be set to True for KD losses"
        featurizer = create_featurizer(task_config.featurizer, task_config.features)
        # load data
        data_handler = create_data_handler(
            task_config.data_handler,
            task_config.features,
            task_config.labels,
            featurizer=featurizer,
        )
        print("\nLoading data...")
        if metadata:
            data_handler.load_metadata(metadata)
        else:
            data_handler.init_metadata()

        metadata = data_handler.metadata

        model = create_model(task_config.model, task_config.features, metadata)

        if lazy.is_lazy(model):
            # finalize any lazy modules before loading weights
            inputs, _, _ = next(iter(data_handler.get_train_iter()))
            lazy.init_lazy_modules(model, inputs)

        if model_state:
            model.load_state_dict(model_state)

        if cuda.CUDA_ENABLED:
            model = model.cuda()
        metric_reporter = create_metric_reporter(task_config.metric_reporter, metadata)
        exporter = (
            create_exporter(
                task_config.exporter,
                task_config.features,
                task_config.labels,
                data_handler.metadata,
                task_config.model,
                task_config.featurizer,
            )
            if task_config.exporter
            else None
        )
        return cls(
            trainer=create_trainer(task_config.trainer, model),
            data_handler=data_handler,
            model=model,
            metric_reporter=metric_reporter,
            exporter=exporter,
        )

    def __init__(
        self,
        trainer: Trainer,
        data_handler: DataHandler,
        model: Model,
        metric_reporter: MetricReporter,
        exporter: Optional[ModelExporter],
    ) -> None:
        self.trainer: Trainer = trainer
        self.data_handler: DataHandler = data_handler
        self.model: Model = model
        self.metric_reporter: MetricReporter = metric_reporter
        self.exporter = exporter

    def train(self, train_config, rank=0, world_size=1, training_state=None):
        """
        Wrapper method to train the model using :class:`~Trainer` object.

        Args:
            train_config (PyTextConfig): config for training
            rank (int): for distributed training only, rank of the gpu, default is 0
            world_size (int): for distributed training only, total gpu to use, default
                is 1
        """
        if training_state:
            result = self.trainer.train_from_state(
                training_state,
                self.data_handler.get_train_iter(rank, world_size),
                self.data_handler.get_eval_iter(),
                self.metric_reporter,
                train_config,
            )
        else:
            result = self.trainer.train(
                self.data_handler.get_train_iter(rank, world_size),
                self.data_handler.get_eval_iter(),
                self.model,
                self.metric_reporter,
                train_config,
                rank=rank,
            )
        return result

    def test(self, test_path):
        """
        Wrapper method to compute test metrics on holdout blind test dataset.

        Args:
            test_path (str): test data file path
        """
        self.data_handler.test_path = test_path
        test_iter = self.data_handler.get_test_iter()
        return self.trainer.test(test_iter, self.model, self.metric_reporter)

    def export(self, model, export_path, metric_channels=None, export_onnx_path=None):
        """
        Wrapper method to export PyTorch model to Caffe2 model using :class:`~Exporter`.

        Args:
            export_path (str): file path of exported caffe2 model
            metric_channels (List[Channel]): outputs of model's execution graph
            export_onnx_path (str):file path of exported onnx model
        """
        # Make sure to put the model on CPU and disable CUDA before exporting to
        # ONNX to disable any data_parallel pieces
        cuda.CUDA_ENABLED = False
        model = model.cpu()
        optimizer = self.trainer.optimizer
        optimizer.pre_export(model)

        if self.exporter:
            if metric_channels:
                print("Exporting metrics")
                self.exporter.export_to_metrics(model, metric_channels)
            print("Saving caffe2 model to: " + export_path)
            self.exporter.export_to_caffe2(model, export_path, export_onnx_path)

    @classmethod
    def format_prediction(cls, predictions, scores, context, target_meta):
        """
        Format the prediction and score from model output, by default just return
        them in a dict
        """
        for prediction, score in zip(predictions, scores):
            yield {"prediction": prediction, "score": score}

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
        model_inputs, context = self.data_handler.get_predict_iter(examples)
        predictions, scores = self.model.get_pred(self.model(*model_inputs))
        results: List = [None] * len(predictions)
        # rearrange to orignal order
        for idx, result in zip(
            context[BatchContext.INDEX],
            self.format_prediction(
                predictions, scores, context, self.data_handler.metadata.target
            ),
        ):
            results[idx] = result
        return results


class Task_Deprecated(TaskBase):
    __EXPANSIBLE__ = True

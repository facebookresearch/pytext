#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from pprint import pprint
from typing import List, Optional

import torch
from pytext.config import ConfigBase, config_to_json
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
from pytext.config.pytext_config import OptimizerParams
from pytext.data import DataHandler
from pytext.data.featurizer import Featurizer, SimpleFeaturizer
from pytext.exporters import ModelExporter
from pytext.metric_reporters import MetricReporter
from pytext.models import Model
from pytext.optimizer import create_optimizer
from pytext.optimizer.scheduler import Scheduler
from pytext.trainers import Trainer
from pytext.utils import cuda_utils


def create_task(task_config, metadata=None, model_state=None):
    return create_component(ComponentType.TASK, task_config, metadata, model_state)


class Task(Component):
    __EXPANSIBLE__ = True
    __COMPONENT_TYPE__ = ComponentType.TASK

    class Config(ConfigBase):
        features: FeatureConfig = FeatureConfig()
        featurizer: Featurizer.Config = SimpleFeaturizer.Config()
        data_handler: DataHandler.Config
        trainer: Trainer.Config = Trainer.Config()
        optimizer: OptimizerParams = OptimizerParams()
        scheduler: Optional[Scheduler.Config] = Scheduler.Config()
        exporter: Optional[ModelExporter.Config] = None

    @classmethod
    def from_config(cls, task_config, metadata=None, model_state=None):
        print("Task parameters:\n")
        pprint(config_to_json(type(task_config), task_config))
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
        if model_state:
            model.load_state_dict(model_state)
        if cuda_utils.CUDA_ENABLED:
            model = model.cuda()
        metric_reporter = create_metric_reporter(task_config.metric_reporter, metadata)
        optimizers = create_optimizer(model, task_config.optimizer)
        exporter = (
            create_exporter(
                task_config.exporter,
                task_config.features,
                task_config.labels,
                data_handler.metadata,
                task_config.model,
            )
            if task_config.exporter
            else None
        )
        return cls(
            trainer=create_trainer(task_config.trainer),
            data_handler=data_handler,
            model=model,
            metric_reporter=metric_reporter,
            optimizers=optimizers,
            lr_scheduler=Scheduler(
                optimizers, task_config.scheduler, metric_reporter.lower_is_better
            ),
            exporter=exporter,
        )

    def __init__(
        self,
        trainer: Trainer,
        data_handler: DataHandler,
        model: Model,
        metric_reporter: MetricReporter,
        optimizers: List[torch.optim.Optimizer],
        lr_scheduler: List[torch.optim.lr_scheduler._LRScheduler],
        exporter: Optional[ModelExporter],
    ) -> None:
        self.trainer: Trainer = trainer
        self.data_handler: DataHandler = data_handler
        self.model: Model = model
        self.metric_reporter: MetricReporter = metric_reporter
        self.optimizers: List[torch.optim.Optimizer] = optimizers
        self.lr_scheduler: List[torch.optim.lr_scheduler._LRScheduler] = lr_scheduler
        self.exporter = exporter

    def train(self, train_config, rank=0, world_size=1):
        return self.trainer.train(
            self.data_handler.get_train_iter(rank, world_size),
            self.data_handler.get_eval_iter(),
            self.model,
            self.metric_reporter,
            train_config,
            self.optimizers,
            self.lr_scheduler,
        )

    def test(self, test_path):
        self.data_handler.test_path = test_path
        test_iter = self.data_handler.get_test_iter()
        return self.trainer.test(test_iter, self.model, self.metric_reporter)

    def export(self, model, export_path, summary_writer=None):
        # Make sure to put the model on CPU and disable CUDA before exporting to
        # ONNX to disable any data_parallel pieces
        cuda_utils.CUDA_ENABLED = False
        model = model.cpu()
        if self.exporter:
            if summary_writer is not None:
                summary_writer.add_graph(model, self.exporter.dummy_model_input)
            print("Saving caffe2 model to: " + export_path)
            self.exporter.export_to_caffe2(model, export_path)

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from collections import OrderedDict
from pprint import pprint
from typing import Dict, Optional

from pytext.config import config_to_json
from pytext.config.component import (
    create_data_handler,
    create_exporter,
    create_featurizer,
    create_metric_reporter,
    create_model,
    create_optimizer,
    create_trainer,
)
from pytext.data import DisjointMultitaskDataHandler
from pytext.metric_reporters.disjoint_multitask_metric_reporter import (
    DisjointMultitaskMetricReporter,
)
from pytext.models.disjoint_multitask_model import DisjointMultitaskModel
from pytext.optimizer.scheduler import Scheduler
from pytext.utils import cuda_utils

from . import Task, TaskBase


class DisjointMultitask(TaskBase):
    """Modules which have the same shared_module_key and type share parameters.
       Only the first instance of such module should be configured in tasks list.
    """

    class Config(TaskBase.Config):
        tasks: Dict[str, Task.Config]
        task_weights: Dict[str, float] = {}
        target_task_name: Optional[str] = None  # for selecting best epoch
        data_handler: DisjointMultitaskDataHandler.Config = DisjointMultitaskDataHandler.Config()
        metric_reporter: DisjointMultitaskMetricReporter.Config = DisjointMultitaskMetricReporter.Config()

    @classmethod
    def from_config(cls, task_config, metadata=None, model_state=None):
        print("Task parameters:\n")
        pprint(config_to_json(type(task_config), task_config))

        data_handlers = OrderedDict()
        exporters = OrderedDict()
        for name, task in task_config.tasks.items():
            featurizer = create_featurizer(task.featurizer, task.features)
            data_handlers[name] = create_data_handler(
                task.data_handler, task.features, task.labels, featurizer=featurizer
            )
        data_handler = DisjointMultitaskDataHandler(
            task_config.data_handler,
            data_handlers,
            target_task_name=task_config.target_task_name,
        )
        print("\nLoading data...")
        if metadata:
            data_handler.load_metadata(metadata)
        else:
            data_handler.init_metadata()
        metadata = data_handler.metadata
        exporters = {
            name: (
                create_exporter(
                    task.exporter,
                    task.features,
                    task.labels,
                    data_handler.data_handlers[name].metadata,
                    task.model,
                )
                if task.exporter
                else None
            )
            for name, task in task_config.tasks.items()
        }
        task_weights = {
            task_name: task_config.task_weights.get(task_name, 1)
            for task_name in task_config.tasks.keys()
        }
        metric_reporter = DisjointMultitaskMetricReporter(
            OrderedDict(
                (name, create_metric_reporter(task.metric_reporter, metadata[name]))
                for name, task in task_config.tasks.items()
            ),
            loss_weights=task_weights,
            target_task_name=task_config.target_task_name,
        )

        model = DisjointMultitaskModel(
            OrderedDict(
                (name, create_model(task.model, task.features, metadata[name]))
                for name, task in task_config.tasks.items()
            ),
            loss_weights=task_weights,
        )
        if model_state:
            model.load_state_dict(model_state)
        if cuda_utils.CUDA_ENABLED:
            model = model.cuda()

        optimizer = create_optimizer(task_config.optimizer, model)
        return cls(
            exporters=exporters,
            trainer=create_trainer(task_config.trainer),
            data_handler=data_handler,
            model=model,
            metric_reporter=metric_reporter,
            optimizer=optimizer,
            lr_scheduler=Scheduler(
                optimizer, task_config.scheduler, metric_reporter.lower_is_better
            ),
        )

    def __init__(self, exporters, **kwargs):
        super().__init__(exporter=None, **kwargs)
        self.exporters = exporters

    def export(
        self, multitask_model, export_path, summary_writer=None, export_onnx_path=None
    ):
        """
        Wrapper method to export PyTorch model to Caffe2 model using :class:`~Exporter`.

        Args:
            export_path (str): file path of exported caffe2 model
            summary_writer: TensorBoard SummaryWriter, used to output the PyTorch
                model's execution graph to TensorBoard, default is None.
            export_onnx_path (str):file path of exported onnx model
        """
        # Make sure to put the model on CPU and disable CUDA before exporting to
        # ONNX to disable any data_parallel pieces
        cuda_utils.CUDA_ENABLED = False
        for name, model in multitask_model.models.items():
            model = model.cpu()
            if self.exporters[name]:
                if summary_writer is not None:
                    self.exporters[name].export_to_tensorboard(model, summary_writer)
                model_export_path = f"{export_path}-{name}"
                model_export_onnx_path = (
                    f"{export_onnx_path}-{name}" if export_onnx_path else None
                )
                print("Saving caffe2 model to: " + model_export_path)
                self.exporters[name].export_to_caffe2(
                    model, model_export_path, model_export_onnx_path
                )

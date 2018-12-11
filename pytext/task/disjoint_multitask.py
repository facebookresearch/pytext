#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from collections import OrderedDict
from pprint import pprint
from typing import Dict

from pytext.config import config_to_json
from pytext.config.component import (
    create_data_handler,
    create_featurizer,
    create_metric_reporter,
    create_model,
    create_trainer,
)
from pytext.data import DisjointMultitaskDataHandler
from pytext.metric_reporters.disjoint_multitask_metric_reporter import (
    DisjointMultitaskMetricReporter,
)
from pytext.models.disjoint_multitask_model import DisjointMultitaskModel
from pytext.optimizer import create_optimizer
from pytext.optimizer.scheduler import Scheduler
from pytext.utils import cuda_utils

from . import Task, TaskBase


class DisjointMultitask(TaskBase):
    """Modules which have the same shared_module_key and type share parameters.
       Only the first instance of such module should be configured in tasks list.
    """

    class Config(TaskBase.Config):
        tasks: Dict[str, Task.Config]
        data_handler: DisjointMultitaskDataHandler.Config = DisjointMultitaskDataHandler.Config()
        metric_reporter: DisjointMultitaskMetricReporter.Config = DisjointMultitaskMetricReporter.Config()

    @classmethod
    def from_config(cls, task_config, metadata=None, model_state=None):
        print("Task parameters:\n")
        pprint(config_to_json(type(task_config), task_config))

        data_handlers = OrderedDict()
        for name, task in task_config.tasks.items():
            featurizer = create_featurizer(task.featurizer, task.features)
            data_handlers[name] = create_data_handler(
                task.data_handler, task.features, task.labels, featurizer=featurizer
            )
        data_handler = DisjointMultitaskDataHandler(
            task_config.data_handler, data_handlers
        )
        print("\nLoading data...")
        if metadata:
            data_handler.load_metadata(metadata)
        else:
            data_handler.init_metadata()
        metadata = data_handler.metadata

        metric_reporter = DisjointMultitaskMetricReporter(
            OrderedDict(
                (name, create_metric_reporter(task.metric_reporter, metadata[name]))
                for name, task in task_config.tasks.items()
            )
        )

        model = DisjointMultitaskModel(
            OrderedDict(
                (name, create_model(task.model, task.features, metadata[name]))
                for name, task in task_config.tasks.items()
            )
        )
        if model_state:
            model.load_state_dict(model_state)
        if cuda_utils.CUDA_ENABLED:
            model = model.cuda()

        optimizers = create_optimizer(model, task_config.optimizer)
        return cls(
            trainer=create_trainer(task_config.trainer),
            data_handler=data_handler,
            model=model,
            metric_reporter=metric_reporter,
            optimizers=optimizers,
            lr_scheduler=Scheduler(
                optimizers, task_config.scheduler, metric_reporter.lower_is_better
            ),
            exporter=None,
        )

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from pytext.metric_reporters.channel import ConsoleChannel, TensorBoardChannel
from pytext.task.new_task import NewTask

from .metric import MyTaggingMetricReporter
from .model import MyTaggingModel


class MyTaggingTask(NewTask):
    class Config(NewTask.Config):
        model: MyTaggingModel.Config = MyTaggingModel.Config()
        metric_reporter: MyTaggingMetricReporter.Config = MyTaggingMetricReporter.Config()

    @classmethod
    def create_metric_reporter(cls, config, tensorizers):
        return MyTaggingMetricReporter(
            channels=[ConsoleChannel(), TensorBoardChannel()],
            label_names=list(tensorizers["slots"].vocab),
        )

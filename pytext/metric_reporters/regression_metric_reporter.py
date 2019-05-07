#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from pytext.metrics import compute_regression_metrics

from .channel import ConsoleChannel
from .metric_reporter import MetricReporter


class RegressionMetricReporter(MetricReporter):

    lower_is_better = False

    class Config(MetricReporter.Config):
        pass

    @classmethod
    def from_config(cls, config, tensorizers=None):
        return cls([ConsoleChannel()])

    def calculate_metric(self):
        assert len(self.all_preds) == len(self.all_targets)
        return compute_regression_metrics(self.all_preds, self.all_targets)

    def get_model_select_metric(self, metrics):
        return metrics.pearson_correlation

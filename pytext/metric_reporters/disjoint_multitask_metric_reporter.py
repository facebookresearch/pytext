#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from pytext.common.constants import BatchContext

from .metric_reporter import MetricReporter


class DisjointMultitaskMetricReporter(MetricReporter):
    def __init__(self, reporters) -> None:
        super().__init__(None)
        self.reporters = reporters

    def add_batch_stats(
        self, n_batches, preds, targets, scores, loss, m_input, **context
    ):
        reporter = self.reporters[context[BatchContext.TASK_NAME]]
        reporter.add_batch_stats(
            n_batches, preds, targets, scores, loss, m_input, **context
        )

    def add_channel(self, channel):
        for reporter in self.reporters.values():
            reporter.add_channel(channel)

    def report_metric(self, stage, epoch, reset=True, print_to_channels=True):
        metrics_dict = {}
        for name, reporter in self.reporters.items():
            print(f"Reporting on task: {name}")
            metrics_dict[name] = reporter.report_metric(
                stage, epoch, reset, print_to_channels
            )
        return metrics_dict

    def compare_metric(self, new_metric, old_metric):
        return True

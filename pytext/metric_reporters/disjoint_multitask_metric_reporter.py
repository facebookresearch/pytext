#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Dict, Optional

from pytext.common.constants import BatchContext

from .metric_reporter import MetricReporter


class DisjointMultitaskMetricReporter(MetricReporter):
    def __init__(
        self,
        reporters: Dict[str, MetricReporter],
        loss_weights: Dict[str, float],
        target_task_name: Optional[str],
    ) -> None:
        """Short summary.

        Args:
            reporters (Dict[str, MetricReporter]):
                Dictionary of sub-task metric-reporters.
            target_task_name (Optional[str]):
                Dev metric for this task will be used to select best epoch.

        Returns:
            None: Description of returned object.

        """

        super().__init__(None)
        self.reporters = reporters
        self.target_task_name = target_task_name or ""
        self.target_reporter = self.reporters.get(self.target_task_name, None)
        self.loss_weights = loss_weights

    def add_batch_stats(
        self, n_batches, preds, targets, scores, loss, m_input, **context
    ):
        # losses are weighted in DisjointMultitaskModel. Here we undo the
        # weighting for proper reporting.
        if self.loss_weights[context[BatchContext.TASK_NAME]] != 0:
            loss /= self.loss_weights[context[BatchContext.TASK_NAME]]
        reporter = self.reporters[context[BatchContext.TASK_NAME]]
        reporter.add_batch_stats(
            n_batches, preds, targets, scores, loss, m_input, **context
        )

    def add_channel(self, channel):
        for reporter in self.reporters.values():
            reporter.add_channel(channel)

    def compare_metric(self, new_metric, old_metric):
        if self.target_reporter:
            return self.target_reporter.compare_metric(new_metric, old_metric)
        else:
            return True

    def report_metric(self, stage, epoch, reset=True, print_to_channels=True):
        metrics_dict = {}
        for name, reporter in self.reporters.items():
            print(f"Reporting on task: {name}")
            metrics_dict[name] = reporter.report_metric(
                stage, epoch, reset, print_to_channels
            )
        if self.target_reporter:
            return metrics_dict[self.target_task_name]
        else:
            return metrics_dict

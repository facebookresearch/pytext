#!/usr/bin/env python3
from typing import Dict, List

import torch
from pytext.config.component import Component, ComponentType
from pytext.config.pytext_config import ConfigBase


class MetricReporter(Component):
    __COMPONENT_TYPE__ = ComponentType.METRIC_REPORTER

    # Whether a lower metric indicates better performance. Set to True for e.g.
    # perplexity, and False for e.g. accuracy.
    lower_is_better: bool = False

    class Config(ConfigBase):
        output_path: str = "/tmp/test_out.txt"

    def __init__(self, channels) -> None:
        self._reset()
        self.channels = channels

    def _reset(self):
        self.all_preds: List = []
        self.all_targets: List = []
        self.all_context: Dict = {}
        self.all_loss: List = []
        self.all_scores: List = []
        self.n_batches = 0
        self.batch_size: List = []

    def add_batch_stats(
        self, n_batches, preds, targets, scores, loss, m_input, **context
    ):
        self.n_batches = n_batches
        self.aggregate_preds(preds)
        self.aggregate_targets(targets)
        self.aggregate_scores(scores)
        for key, val in context.items():
            if not (isinstance(val, torch.Tensor) or isinstance(val, List)):
                continue
            if key not in self.all_context:
                self.all_context[key] = []
            self.aggregate_data(self.all_context[key], val)
        self.all_loss.append(loss)
        self.batch_size.append(len(targets))

    def aggregate_preds(self, new_batch):
        self.aggregate_data(self.all_preds, new_batch)

    def aggregate_targets(self, new_batch):
        self.aggregate_data(self.all_targets, new_batch)

    def aggregate_scores(self, new_batch):
        self.aggregate_data(self.all_scores, new_batch)

    @classmethod
    def aggregate_data(cls, all_data, new_batch):
        if new_batch is None:
            return
        simple_list = cls._make_simple_list(new_batch)
        all_data.extend(simple_list)

    @classmethod
    def _make_simple_list(cls, data):
        if isinstance(data, torch.Tensor):
            return data.tolist()
        elif isinstance(data, List) and all(
            isinstance(elem, torch.Tensor) for elem in data
        ):
            return [elem.tolist() for elem in data]
        elif isinstance(data, List):
            return data
        else:
            raise NotImplementedError()

    def calculate_loss(self):
        return sum(self.all_loss) / float(len(self.all_loss))

    def calculate_metric(self):
        raise NotImplementedError()

    def gen_extra_context(self):
        pass

    def get_meta(self):
        return {}

    def report_metric(self, stage, epoch, reset=True):
        self.gen_extra_context()
        self.total_loss = self.calculate_loss()
        metrics = self.calculate_metric()
        model_select_metric = self.get_model_select_metric(metrics)

        for channel in self.channels:
            if stage in channel.stages:
                channel.report(
                    stage,
                    epoch,
                    metrics,
                    model_select_metric,
                    self.total_loss,
                    self.all_preds,
                    self.all_targets,
                    self.all_scores,
                    self.all_context,
                    self.get_meta(),
                )

        if reset:
            self._reset()
        return metrics

    @staticmethod
    def get_model_select_metric(metrics):
        return metrics

    @classmethod
    def compare_metric(cls, new_metric, old_metric):
        """return True if new metric indicates better model performance
        """
        if not old_metric:
            return True

        new = cls.get_model_select_metric(new_metric)
        old = cls.get_model_select_metric(old_metric)
        if new == old:
            return False
        return (new < old) == cls.lower_is_better

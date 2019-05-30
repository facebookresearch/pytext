#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Dict, List

import numpy as np
import torch
from pytext.common.constants import DatasetFieldName
from pytext.config.component import Component, ComponentType
from pytext.config.pytext_config import ConfigBase
from pytext.metrics import RealtimeMetrics
from pytext.utils.meter import TimeMeter


class MetricReporter(Component):
    """
    MetricReporter is responsible of three things:

    #. Aggregate output from trainer, which includes model inputs, predictions,
       targets, scores, and loss.
    #. Calculate metrics using the aggregated output, and define how the metric
       is used to find best model
    #. Optionally report the metrics and aggregated output to various channels

    Attributes:
        lower_is_better (bool): Whether a lower metric indicates better performance.
            Set to True for e.g. perplexity, and False for e.g. accuracy. Default
            is False
        channels (List[Channel]): A list of Channel that will receive metrics and
            the aggregated trainer output then format and report them in any customized
            way.

    MetricReporter is tightly-coupled with metric aggregation and computation which
    makes inheritance hard to reuse the parent functionalities and attributes. Next
    step is to decouple the metric aggregation and computation vs metric reporting.
    """

    __COMPONENT_TYPE__ = ComponentType.METRIC_REPORTER

    lower_is_better: bool = False
    realtime_report_freq: int = 500

    class Config(ConfigBase):
        output_path: str = "/tmp/test_out.txt"

    def __init__(self, channels) -> None:
        self._reset()
        self.channels = channels
        self._reset_realtime()

    def _reset(self):
        self.all_preds: List = []
        self.all_targets: List = []
        self.all_context: Dict = {}
        self.all_loss: List = []
        self.all_scores: List = []
        self.n_batches = 0
        self.batch_size: List = []

    def _reset_realtime(self):
        self.realtime_meters: Dict = {}
        self.realtime_meters["tps"] = TimeMeter()  # tokens per second
        self.realtime_meters["ups"] = TimeMeter()  # updates per second

    def add_batch_stats(
        self, n_batches, preds, targets, scores, loss, m_input, **context
    ):
        """
        Aggregates a batch of output data (predictions, scores, targets/true labels
        and loss).

        Args:
            n_batches (int): number of current batch
            preds (torch.Tensor): predictions of current batch
            targets (torch.Tensor): targets of current batch
            scores (torch.Tensor): scores of current batch
            loss (double): average loss of current batch
            m_input (Tuple[torch.Tensor, ...]): model inputs of current batch
            context (Dict[str, Any]): any additional context data, it could be
                either a list of data which maps to each example, or a single value
                for the batch
        """
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
        # some loss functions (eg: in NewBertRegressionTask) return a tensor
        # convert tensor to float
        self.all_loss.append(float(loss))
        self.batch_size.append(len(m_input[0]))

        # realtime stats
        if DatasetFieldName.NUM_TOKENS in context:
            self.realtime_meters["tps"].update(context[DatasetFieldName.NUM_TOKENS])
            self.realtime_meters["ups"].update(1)
        if n_batches and n_batches % self.realtime_report_freq == 0:
            self.report_realtime_metric()

    def aggregate_preds(self, new_batch):
        self.aggregate_data(self.all_preds, new_batch)

    def aggregate_targets(self, new_batch):
        self.aggregate_data(self.all_targets, new_batch)

    def aggregate_scores(self, new_batch):
        self.aggregate_data(self.all_scores, new_batch)

    @classmethod
    def aggregate_data(cls, all_data, new_batch):
        """
        Aggregate a batch of data, basically just convert tensors to list of native
        python data
        """
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
        elif (
            isinstance(data, List)
            and all(isinstance(elem, List) for elem in data)
            and all(
                isinstance(elem, torch.Tensor) for elemlist in data for elem in elemlist
            )
        ):
            return [[elem.tolist() for elem in elemlist] for elemlist in data]
        elif isinstance(data, List):
            return data
        elif isinstance(data, tuple):
            return data[0].tolist()
        else:
            raise NotImplementedError()

    def add_channel(self, channel):
        self.channels.append(channel)

    def batch_context(self, raw_batch, batch):
        context = {}
        if DatasetFieldName.NUM_TOKENS in batch:
            context.update(
                {DatasetFieldName.NUM_TOKENS: batch[DatasetFieldName.NUM_TOKENS]}
            )
        return context

    def calculate_loss(self):
        """
        Calculate the average loss for all aggregated batch
        """
        return np.average(self.all_loss, weights=self.batch_size)

    def calculate_metric(self):
        """
        Calculate metrics, each sub class should implement it
        """
        raise NotImplementedError()

    def gen_extra_context(self):
        """
        Generate any extra intermediate context data for metric calculation
        """
        pass

    # TODO this method can be removed by moving Channel construction to Task
    def get_meta(self):
        """
        Get global meta data that is not specific to any batch, the data will be
        pass along to channels
        """
        return {}

    def report_metric(self, model, stage, epoch, reset=True, print_to_channels=True):
        """
        Calculate metrics and average loss, report all statistic data to channels

        Args:
            model (nn.Module): the PyTorch neural network model.
            stage (Stage): training, evaluation or test
            epoch (int): current epoch
            reset (bool): if all data should be reset after report, default is True
            print_to_channels (bool): if report data to channels, default is True
        """
        self.gen_extra_context()
        self.total_loss = self.calculate_loss()
        metrics = self.calculate_metric()
        model_select_metric = self.get_model_select_metric(metrics)

        if print_to_channels:
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
                        model,
                    )
            self.report_realtime_metric()

        if reset:
            self._reset()
            self._reset_realtime()
        return metrics

    def report_realtime_metric(self):
        tps = self.realtime_meters["tps"].avg
        ups = self.realtime_meters["ups"].avg

        if not tps or not ups:
            return
        metrics = RealtimeMetrics(samples=self.n_batches + 1, tps=tps, ups=ups)
        print(metrics, flush=True)

    def get_model_select_metric(self, metrics):
        """
        Return a single numeric metric value that is used for model selection, returns
        the metric itself by default, but usually metrics will be more complicated
        data structures
        """
        return metrics

    def compare_metric(self, new_metric, old_metric):
        """
        Check if new metric indicates better model performance

        Returns:
            bool, true if model with new_metric performs better
        """
        if not old_metric:
            return True

        new = self.get_model_select_metric(new_metric)
        old = self.get_model_select_metric(old_metric)
        if new == old:
            return False
        return (new < old) == self.lower_is_better

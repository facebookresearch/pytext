#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Dict, List

import torch
from pytext.config.component import Component, ComponentType
from pytext.config.pytext_config import ConfigBase


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
    """

    __COMPONENT_TYPE__ = ComponentType.METRIC_REPORTER

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
        """
        Aggregate a batch of data, bascically just convert tensors to list of native
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
        elif isinstance(data, List):
            return data
        else:
            raise NotImplementedError()

    def add_channel(self, channel):
        self.channels.append(channel)

    def calculate_loss(self):
        """
        Calculate the average loss for all aggregated batch
        """
        return sum(self.all_loss) / float(len(self.all_loss))

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

    def report_metric(self, stage, epoch, reset=True, print_to_channels=True):
        """
        Calculate metrics and average loss, report all statistic data to channels

        Args:
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
                    )

        if reset:
            self._reset()
        return metrics

    @staticmethod
    def get_model_select_metric(metrics):
        """
        Return a single numeric metric value that is used for model selection, returns
        the metric itself by default, but usually metrics will be more complicated
        data structures
        """
        return metrics

    @classmethod
    def compare_metric(cls, new_metric, old_metric):
        """
        Check if new metric indicates better model performance

        Returns:
            bool, true if model with new_metric performs better
        """
        if not old_metric:
            return True

        new = cls.get_model_select_metric(new_metric)
        old = cls.get_model_select_metric(old_metric)
        if new == old:
            return False
        return (new < old) == cls.lower_is_better

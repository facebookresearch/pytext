#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from pytext.data import CommonMetadata
from pytext.metrics import compute_pairwise_ranking_metrics

from .channel import ConsoleChannel
from .metric_reporter import MetricReporter


class PairwiseRankingMetricReporter(MetricReporter):
    @classmethod
    def from_config(cls, config, meta: CommonMetadata = None, tensorizers=None):
        # TODO: add file channel
        return cls([ConsoleChannel()])

    def calculate_metric(self):
        return compute_pairwise_ranking_metrics(self.all_preds, self.all_scores)

    def add_batch_stats(
        self, n_batches, preds, targets, scores, loss, m_input, **context
    ):
        # target = 1 means the first response was ranked higher than the second response
        # however, our training data is tuples of {pos_response, neg_response} pairs
        # i.e, pos_response is always the first response, neg_response is always the
        # second response. so target = 1 for all cases
        targets = [1] * preds.shape[0]
        super().add_batch_stats(
            n_batches, preds, targets, scores, loss, m_input, **context
        )

    @staticmethod
    def get_model_select_metric(metrics):
        return metrics.accuracy

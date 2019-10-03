#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import itertools

from pytext.metric_reporters.channel import ConsoleChannel, TensorBoardChannel
from pytext.metric_reporters.metric_reporter import MetricReporter
from pytext.metrics import LabelPrediction, compute_classification_metrics


class MyTaggingMetricReporter(MetricReporter):
    @classmethod
    def from_config0(cls, config, vocab):
        return MyTaggingMetricReporter(
            channels=[ConsoleChannel(), TensorBoardChannel()], label_names=vocab
        )

    @classmethod
    def from_config(cls, config, tensorizers):
        return MyTaggingMetricReporter(
            channels=[ConsoleChannel(), TensorBoardChannel()],
            label_names=tensorizers["slots"].vocab,
        )

    def __init__(self, label_names, channels):
        super().__init__(channels)
        self.label_names = label_names

    def calculate_metric(self):
        return compute_classification_metrics(
            list(
                itertools.chain.from_iterable(
                    (LabelPrediction(s, p, e) for s, p, e in zip(scores, pred, expect))
                    for scores, pred, expect in zip(
                        self.all_scores, self.all_preds, self.all_targets
                    )
                )
            ),
            self.label_names,
            self.calculate_loss(),
        )

    # def batch_context(self, batch):
    #    return {}

    @staticmethod
    def get_model_select_metric(metrics):
        return metrics.accuracy

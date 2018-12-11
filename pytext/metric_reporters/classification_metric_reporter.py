#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List

from pytext.common.constants import Stage
from pytext.data import CommonMetadata
from pytext.metrics import LabelPrediction, compute_classification_metrics

from .channel import Channel, ConsoleChannel, FileChannel
from .metric_reporter import MetricReporter


class IntentModelChannel(FileChannel):
    def get_title(self):
        return ("predicted", "actual", "scores_str", "text")

    def gen_content(self, metrics, loss, preds, targets, scores, contexts):
        for i in range(len(preds)):
            yield [
                preds[i],
                targets[i],
                ",".join([f"{s:.2f}" for s in scores[i]]),
                contexts["utterance"][i],
            ]


class ClassificationMetricReporter(MetricReporter):
    def __init__(self, label_names: List[str], channels: List[Channel]) -> None:
        super().__init__(channels)
        self.label_names = label_names

    @classmethod
    def from_config(cls, config, meta: CommonMetadata):
        label_names = meta.target.vocab.itos
        return cls(
            label_names,
            [ConsoleChannel(), IntentModelChannel((Stage.TEST,), config.output_path)],
        )

    def calculate_metric(self):
        return compute_classification_metrics(
            [
                LabelPrediction(scores, pred, expect)
                for scores, pred, expect in zip(
                    self.all_scores, self.all_preds, self.all_targets
                )
            ],
            self.label_names,
        )

    def get_meta(self):
        return {"label_names": self.label_names}

    @staticmethod
    def get_model_select_metric(metrics):
        return metrics.accuracy

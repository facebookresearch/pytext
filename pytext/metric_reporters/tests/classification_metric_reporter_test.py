#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from unittest import TestCase

from pytext.metric_reporters.classification_metric_reporter import (
    compute_topk_classification_metrics,
)
from pytext.metrics import ClassificationMetrics, LabelTopKPrediction


class TestClassificationMetricReporter(TestCase):
    test_labels = ["hello", "hi", "hey", "how are you"]

    def test_compute_topk_classification_metrics_zero_correct(self):
        metrics: ClassificationMetrics = compute_topk_classification_metrics(
            predictions=[
                LabelTopKPrediction([0.5, 0.3, 0.2], [0, 1, 2], 3),
                LabelTopKPrediction([0.5, 0.3, 0.2], [0, 2, 3], 1),
            ],
            label_names=self.test_labels,
            loss=0,
        )

        self.assertEqual(0, metrics.accuracy)

    def test_compute_topk_classification_metrics_two_thirds_correct(self):
        metrics: ClassificationMetrics = compute_topk_classification_metrics(
            predictions=[
                LabelTopKPrediction([0.5, 0.3, 0.2], [0, 1, 2], 0),
                LabelTopKPrediction([0.5, 0.3, 0.2], [0, 2, 3], 2),
                LabelTopKPrediction([0.5, 0.3, 0.2], [1, 2, 3], 0),
            ],
            label_names=self.test_labels,
            loss=0,
        )

        self.assertEqual(2 / 3, metrics.accuracy)

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from pytext.metrics import (
    LabelListPrediction,
    compute_multi_label_classification_metrics,
)
from pytext.metrics.tests.metrics_test_base import MetricsTestBase


LABEL_NAMES = ["label1", "label2", "label3"]
PREDICTIONS = [
    LabelListPrediction(scores, predicted, expected)
    for scores, predicted, expected in [
        ([-0.5, -0.7, -0.8], [1, 0, 0], [0]),
        ([-0.9, -0.2, -0.9], [0, 1, 0], [2]),
        ([-0.7, -0.4, -0.7], [0, 1, 0], [1]),
        ([-0.8, -0.9, -0.3], [0, 0, 1], [1]),
    ]
]


class BasicMetricsTest(MetricsTestBase):
    def test_compute_multi_label_classification_metrics(self) -> None:

        roc_auc_dict = {"label1": 1.0, "label2": 0.25, "label3": 0.0}

        metrics = compute_multi_label_classification_metrics(
            PREDICTIONS, LABEL_NAMES, loss=5.0
        )
        self.assertAlmostEqual(metrics.roc_auc, 1.25 / 3)
        for k, v in metrics.per_label_soft_scores.items():
            metric_value = getattr(v, "roc_auc", None)
            self.assertAlmostEqual(metric_value, roc_auc_dict[k])

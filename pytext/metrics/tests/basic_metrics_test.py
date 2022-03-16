#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from pytext.metrics import (
    ClassificationMetrics,
    LabelPrediction,
    MacroPRF1Metrics,
    MacroPRF1Scores,
    PRF1Scores,
    SoftClassificationMetrics,
    compute_classification_metrics,
    compute_soft_metrics,
)
from pytext.metrics.tests.metrics_test_base import MetricsTestBase


LABEL_NAMES1 = ["label1", "label2", "label3"]
PREDICTIONS1 = [
    LabelPrediction(scores, predicted, expected)
    for scores, predicted, expected in [
        ([-0.5, -0.7, -0.8], 0, 0),
        ([-0.9, -0.2, -0.9], 1, 0),
        ([-0.7, -0.4, -0.9], 1, 1),
        ([-0.8, -0.9, -0.3], 2, 1),
    ]
]

LABEL_NAMES2 = ["label1", "label2"]
PREDICTIONS2 = [
    LabelPrediction(scores, predicted, expected)
    for scores, predicted, expected in [
        ([-0.6, -0.4], 1, 0),
        ([-0.7, -0.8], 0, 0),
        ([-0.6, -0.2], 1, 1),
        ([-0.5, -0.8], 0, 1),
        ([-0.6, -0.8], 0, 0),
    ]
]


class BasicMetricsTest(MetricsTestBase):
    def test_prf1_metrics(self) -> None:
        self.assertMetricsAlmostEqual(
            compute_classification_metrics(
                PREDICTIONS1, LABEL_NAMES1, loss=2.0, average_precisions=False
            ),
            ClassificationMetrics(
                accuracy=0.5,
                macro_prf1_metrics=MacroPRF1Metrics(
                    per_label_scores={
                        # label1: TP = 1, FP = 0, FN = 1
                        "label1": PRF1Scores(1, 0, 1, 1.0, 0.5, 2.0 / 3),
                        # label2: TP = 1, FP = 1, FN = 1
                        "label2": PRF1Scores(1, 1, 1, 0.5, 0.5, 0.5),
                        # label3: TP = 0, FP = 1, FN = 0
                        "label3": PRF1Scores(0, 1, 0, 0.0, 0.0, 0.0),
                    },
                    macro_scores=MacroPRF1Scores(3, 0.5, 1.0 / 3, 7.0 / 18),
                ),
                per_label_soft_scores=None,
                mcc=None,
                roc_auc=None,
                loss=2.0,
            ),
        )

    def test_soft_metrics_computation(self) -> None:
        recall_at_precision_dict_l1 = {0.9: 0.0, 0.8: 0.0, 0.6: 1.0, 0.4: 1.0, 0.2: 1.0}
        decision_thresh_at_precision_dict_l1 = {
            0.9: 0.0,
            0.8: 0.0,
            0.6: -0.7,
            0.4: -0.7,
            0.2: -0.7,
        }
        recall_at_precision_dict_l2 = {0.9: 0.5, 0.8: 0.5, 0.6: 0.5, 0.4: 1.0, 0.2: 1.0}
        decision_thresh_at_precision_dict_l2 = {
            0.9: -0.2,
            0.8: -0.2,
            0.6: -0.2,
            0.4: -0.8,
            0.2: -0.8,
        }
        precision_at_recall_dict_l1 = {0.9: 0.6, 0.8: 0.6, 0.6: 0.6, 0.4: 0.6, 0.2: 0.6}
        decision_thresh_at_recall_dict_l1 = {
            0.9: -0.7,
            0.8: -0.7,
            0.6: -0.7,
            0.4: -0.7,
            0.2: -0.7,
        }
        precision_at_recall_dict_l2 = {0.9: 0.4, 0.8: 0.4, 0.6: 0.4, 0.4: 1.0, 0.2: 1.0}
        decision_thresh_at_recall_dict_l2 = {
            0.9: -0.8,
            0.8: -0.8,
            0.6: -0.8,
            0.4: -0.2,
            0.2: -0.2,
        }
        self.assertMetricsAlmostEqual(
            compute_soft_metrics(PREDICTIONS2, LABEL_NAMES2),
            {
                "label1": SoftClassificationMetrics(
                    average_precision=8.0 / 15,
                    recall_at_precision=recall_at_precision_dict_l1,
                    decision_thresh_at_precision=decision_thresh_at_precision_dict_l1,
                    precision_at_recall=precision_at_recall_dict_l1,
                    decision_thresh_at_recall=decision_thresh_at_recall_dict_l1,
                    roc_auc=1.0 / 6,
                ),
                "label2": SoftClassificationMetrics(
                    average_precision=0.7,
                    recall_at_precision=recall_at_precision_dict_l2,
                    decision_thresh_at_precision=decision_thresh_at_precision_dict_l2,
                    precision_at_recall=precision_at_recall_dict_l2,
                    decision_thresh_at_recall=decision_thresh_at_recall_dict_l2,
                    roc_auc=4.0 / 6,
                ),
            },
        )

    def test_compute_mcc(self) -> None:
        metrics = compute_classification_metrics(PREDICTIONS2, LABEL_NAMES2, loss=5.0)
        self.assertAlmostEqual(metrics.mcc, 1.0 / 6)
        # Just to test the metrics print without errors
        metrics.print_metrics()

    def test_compute_roc_auc(self) -> None:
        metrics = compute_classification_metrics(PREDICTIONS2, LABEL_NAMES2, loss=5.0)
        self.assertAlmostEqual(metrics.roc_auc, 1.0 / 6)

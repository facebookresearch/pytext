#!/usr/bin/env python3

from typing import List, Tuple

from pytext.metrics import (
    AllClassificationMetrics,
    ClassificationMetrics,
    MacroClassificationMetrics,
    compute_classification_metrics,
)

from .metrics_test_base import MetricsTestBase


TEST_EXAMPLES: List[Tuple[str, str]] = [
    ("label1", "label1"),
    ("label2", "label1"),
    ("label2", "label2"),
    ("label3", "label2"),
]


class BasicMetricsTest(MetricsTestBase):
    def test_compute_classification_metrics(self) -> None:
        self.assertMetricsAlmostEqual(
            compute_classification_metrics(TEST_EXAMPLES),
            AllClassificationMetrics(
                per_label_scores={
                    # label1: TP = 1, FP = 0, FN = 1
                    "label1": ClassificationMetrics(1, 0, 1, 1.0, 0.5, 2.0 / 3),
                    # label2: TP = 1, FP = 1, FN = 1
                    "label2": ClassificationMetrics(1, 1, 1, 0.5, 0.5, 0.5),
                    # label3: TP = 0, FP = 1, FN = 0
                    "label3": ClassificationMetrics(0, 1, 0, 0.0, 0.0, 0.0),
                },
                macro_scores=MacroClassificationMetrics(3, 0.5, 1.0 / 3, 7.0 / 18),
                # all labels: TP = 2, FP = 2, FN = 2
                micro_scores=ClassificationMetrics(2, 2, 2, 0.5, 0.5, 0.5),
            ),
        )

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import math
from typing import Dict, List, NamedTuple, Tuple

import numpy as np
from pytext.metrics import LabelPrediction


def get_bucket_scores(
    y_score: List[float], buckets: int = 10
) -> Tuple[List[List[float]], List[int]]:
    """
    Organizes real-valued posterior probabilities into buckets. For example, if
    we have 10 buckets, the probabilities 0.0, 0.1, 0.2 are placed into buckets
    0 (0.0 <= p < 0.1), 1 (0.1 <= p < 0.2), and 2 (0.2 <= p < 0.3), respectively.
    """

    bucket_values = [[] for _ in range(buckets)]
    bucket_indices = [[] for _ in range(buckets)]
    for i, score in enumerate(y_score):
        for j in range(buckets):
            if score < float((j + 1) / buckets):
                break
        bucket_values[j].append(score)
        bucket_indices[j].append(i)
    return (bucket_values, bucket_indices)


def get_bucket_confidence(bucket_values: List[List[float]]) -> List[float]:
    """
    Computes average confidence for each bucket. If a bucket does not have any
    predictions, uses -1 as a placeholder.
    """

    return [np.mean(bucket) if len(bucket) > 0 else -1.0 for bucket in bucket_values]


def get_bucket_accuracy(
    bucket_values: List[List[float]], y_true: List[float], y_pred: List[float]
) -> List[float]:
    """
    Computes accuracy for each bucket. If a bucket does not have any predictions,
    uses -1 as a placeholder.
    """

    per_bucket_correct = [
        [int(y_true[i] == y_pred[i]) for i in bucket] for bucket in bucket_values
    ]
    return [
        np.mean(bucket) if len(bucket) > 0 else -1.0 for bucket in per_bucket_correct
    ]


def calculate_error(
    n_samples: int,
    bucket_values: List[List[float]],
    bucket_confidence: List[List[float]],
    bucket_accuracy: List[List[float]],
) -> Tuple[float, float, float]:
    """
    Computes several metrics used to measure calibration error, including
    expected calibration error (ECE), maximum calibration error (MCE), and
    total calibration error (TCE).
    """

    assert len(bucket_values) == len(bucket_confidence) == len(bucket_accuracy)
    assert sum(map(len, bucket_values)) == n_samples

    expected_error, max_error, total_error = 0.0, 0.0, 0.0
    for (bucket, accuracy, confidence) in zip(
        bucket_values, bucket_accuracy, bucket_confidence
    ):
        if len(bucket) > 0:
            delta = abs(accuracy - confidence)
            expected_error += (len(bucket) / n_samples) * delta
            max_error = max(max_error, delta)
            total_error += delta
    return (expected_error * 100.0, max_error * 100.0, total_error * 100.0)


class CalibrationMetrics(NamedTuple):
    expected_error: float
    max_error: float
    total_error: float

    def print_metrics(self, report_pep=False) -> None:
        print(f"\tExpected Error: {self.expected_error * 100.:.2f}")
        print(f"\tMax Error: {self.max_error * 100.:.2f}")
        print(f"\tTotal Error: {self.total_error * 100.:.2f}")


class AllCalibrationMetrics(NamedTuple):
    calibration_metrics: Dict[str, CalibrationMetrics]

    def print_metrics(self, report_pep=False) -> None:
        for (name, calibration_metric) in self.calibration_metrics.items():
            print(f"> {name}")
            calibration_metric.print_metrics()


def compute_calibration(
    label_predictions: List[LabelPrediction],
) -> Tuple[float, float, float]:
    conf_list = [
        math.exp(prediction.label_scores[prediction.predicted_label])  # exp(log(p))
        for prediction in label_predictions
    ]
    pred_list = [prediction.predicted_label for prediction in label_predictions]
    true_list = [prediction.expected_label for prediction in label_predictions]

    bucket_values, bucket_indices = get_bucket_scores(conf_list)
    bucket_confidence = get_bucket_confidence(bucket_values)
    bucket_accuracy = get_bucket_accuracy(bucket_indices, true_list, pred_list)

    expected_error, max_error, total_error = calculate_error(
        n_samples=len(conf_list),
        bucket_values=bucket_values,
        bucket_confidence=bucket_confidence,
        bucket_accuracy=bucket_accuracy,
    )

    return CalibrationMetrics(
        expected_error=expected_error, max_error=max_error, total_error=total_error
    )

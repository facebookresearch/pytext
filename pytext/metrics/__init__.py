#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from collections import defaultdict
from typing import (
    Any,
    DefaultDict,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np


RECALL_AT_PRECISION_THREHOLDS = [0.2, 0.4, 0.6, 0.8, 0.9]

"""
Basic metric classes and functions for single-label prediction problems.
"""


class LabelPrediction(NamedTuple):
    """
    Label predictions of an example.

    Attributes:
        label_scores: Confidence scores that each label receives.
        predicted_label: Index of the predicted label. This is usually the label with
            the highest confidence score in label_scores.
        expected_label: Index of the true label.
    """

    label_scores: List[float]
    predicted_label: int
    expected_label: int


class PRF1Scores(NamedTuple):
    """
    Precision/recall/F1 scores for a collection of predictions.

    Attributes:
        true_positives: Number of true positives.
        false_positives: Number of false positives.
        false_negatives: Number of false negatives.
        precision: TP / (TP + FP).
        recall: TP / (TP + FN).
        f1: 2 * TP / (2 * TP + FP + FN).
    """

    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float


class SoftClassificationMetrics(NamedTuple):
    """
    Classification scores that are independent of thresholds.
    """

    average_precision: float
    recall_at_precision: Dict[float, float]


class MacroPRF1Scores(NamedTuple):
    """
    Macro precision/recall/F1 scores (averages across each label).

    Attributes:
        num_label: Number of distinct labels.
        precision: Equally weighted average of precisions for each label.
        recall: Equally weighted average of recalls for each label.
        f1: Equally weighted average of F1 scores for each label.
    """

    num_labels: int
    precision: float
    recall: float
    f1: float


class MacroPRF1Metrics(NamedTuple):
    """
    Aggregated metric class for macro precision/recall/F1 scores.

    Attributes:
        per_label_scores: Mapping from label string to the corresponding
            precision/recall/F1 scores.
        macro_scores: Macro precision/recall/F1 scores across the labels in
            `per_label_scores`.
    """

    per_label_scores: Dict[str, PRF1Scores]
    macro_scores: MacroPRF1Scores

    def print_metrics(self) -> None:
        res = (
            f"\t{'Label':<20}"
            f"\t{'Precision':<10}"
            f"\t{'Recall':<10}"
            f"\t{'F1':<10}"
            f"\t{'Support':<10}\n\n"
        )
        for label, label_metrics in self.per_label_scores.items():
            support = label_metrics.true_positives + label_metrics.false_negatives
            res += (
                f"\t{label:<20}"
                f"\t{label_metrics.precision * 100:<10.2f}"
                f"\t{label_metrics.recall * 100:<10.2f}"
                f"\t{label_metrics.f1 * 100:<10.2f}"
                f"\t{support:<10}\n"
            )
        res += (
            f"\t{'Overall macro scores':<20}"
            f"\t{self.macro_scores.precision * 100:<10.2f}"
            f"\t{self.macro_scores.recall * 100:<10.2f}"
            f"\t{self.macro_scores.f1 * 100:<10.2f}"
        )
        print(res)


class PRF1Metrics(NamedTuple):
    """
    Metric class for all types of precision/recall/F1 scores.

    Attributes:
        per_label_scores: Map from label string to the corresponding precision/recall/F1
            scores.
        macro_scores: Macro precision/recall/F1 scores across the labels in
            `per_label_scores`.
        micro_scores: Micro (regular) precision/recall/F1 scores for the same
            collection of predictions.
    """

    per_label_scores: Dict[str, PRF1Scores]
    macro_scores: MacroPRF1Scores
    micro_scores: PRF1Scores

    def print_metrics(self) -> None:
        res = (
            f"\t{'Per label scores':<40}"
            f"\t{'Precision':<10}"
            f"\t{'Recall':<10}"
            f"\t{'F1':<10}"
            f"\t{'Support':<10}\n\n"
        )
        for label, label_metrics in self.per_label_scores.items():
            support = label_metrics.true_positives + label_metrics.false_negatives
            res += (
                f"\t{label:<40}"
                f"\t{label_metrics.precision * 100:<10.2f}"
                f"\t{label_metrics.recall * 100:<10.2f}"
                f"\t{label_metrics.f1 * 100:<10.2f}"
                f"\t{support:<10}\n"
            )
        support = self.micro_scores.true_positives + self.micro_scores.false_negatives
        res += (
            f"\n\t{'Overall micro scores':<40}"
            f"\t{self.micro_scores.precision * 100:<10.2f}"
            f"\t{self.micro_scores.recall * 100:<10.2f}"
            f"\t{self.micro_scores.f1 * 100:<10.2f}"
            f"\t{support:<10}\n"
        )
        res += (
            f"\t{'Overall macro scores':<40}"
            f"\t{self.macro_scores.precision * 100:<10.2f}"
            f"\t{self.macro_scores.recall * 100:<10.2f}"
            f"\t{self.macro_scores.f1 * 100:<10.2f}\n"
        )
        print(res)


class ClassificationMetrics(NamedTuple):
    """
    Metric class for various classification metrics.

    Attributes:
        accuracy: Overall accuracy of predictions.
        macro_prf1_metrics: Macro precision/recall/F1 scores.
        per_label_soft_scores: Per label soft metrics.
        mcc: Matthews correlation coefficient.
    """

    accuracy: float
    macro_prf1_metrics: MacroPRF1Metrics
    per_label_soft_scores: Optional[Dict[str, SoftClassificationMetrics]]
    mcc: Optional[float]

    def print_metrics(self) -> None:
        print(f"Accuracy: {self.accuracy * 100:.2f}\n")
        print("Macro P/R/F1 Scores:")
        self.macro_prf1_metrics.print_metrics()
        print("\nSoft Metrics:")
        if self.per_label_soft_scores:
            print(f"\t{'Label':<10}\t{'Average precision':<10}")
            for label, label_metrics in self.per_label_soft_scores.items():
                print(f"\t{label:<10}\t{label_metrics.average_precision * 100:<10.2f}")
            for label, label_metrics in self.per_label_soft_scores.items():
                for threshold, recall in label_metrics.recall_at_precision.items():
                    print(f"\t{'Label':<10}\tRecall at precision {threshold}")
                    print(f"\t{label:<10}\t{recall * 100:<10.2f}")
        if self.mcc:
            print(f"\nMatthews correlation coefficient: {self.mcc :.2f}")


class Confusions:
    """
    Confusion information for a collection of predictions.

    Attributes:
        TP: Number of true positives.
        FP: Number of false positives.
        FN: Number of false negatives.
    """

    __slots__ = "TP", "FP", "FN"

    def __init__(self, TP: int = 0, FP: int = 0, FN: int = 0) -> None:
        self.TP: int = TP
        self.FP: int = FP
        self.FN: int = FN

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Confusions):
            return NotImplemented
        return self.TP == other.TP and self.FP == other.FP and self.FN == other.FN

    def __add__(self, other: "Confusions") -> "Confusions":
        return Confusions(
            TP=self.TP + other.TP, FP=self.FP + other.FP, FN=self.FN + other.FN
        )

    def __iadd__(self, other: "Confusions") -> "Confusions":
        self.TP += other.TP
        self.FP += other.FP
        self.FN += other.FN
        return self

    def _asdict(self) -> Dict:
        return {"TP": self.TP, "FP": self.FP, "FN": self.FN}

    def compute_metrics(self) -> PRF1Scores:
        precision, recall, f1 = compute_prf1(self.TP, self.FP, self.FN)
        return PRF1Scores(
            true_positives=self.TP,
            false_positives=self.FP,
            false_negatives=self.FN,
            precision=precision,
            recall=recall,
            f1=f1,
        )


class PerLabelConfusions:
    """
    Per label confusion information.

    Attributes:
        label_confusions_map: Map from label string to the corresponding confusion
            counts.
    """

    __slots__ = "label_confusions_map"

    def __init__(self) -> None:
        self.label_confusions_map: DefaultDict[str, Confusions] = defaultdict(
            Confusions
        )

    def update(self, label: str, item: str, count: int) -> None:
        """
        Increase one of TP, FP or FN count for a label by certain amount.

        Args:
            label: Label to be modified.
            item: Type of count to be modified, should be one of "TP", "FP" or "FN".
            count: Amount to be added to the count.

        Returns:
            None
        """
        confusions = self.label_confusions_map[label]
        setattr(confusions, item, getattr(confusions, item) + count)

    def compute_metrics(self) -> MacroPRF1Metrics:
        per_label_scores: Dict[str, PRF1Scores] = {}
        precision_sum, recall_sum, f1_sum = 0.0, 0.0, 0.0
        for label, confusions in self.label_confusions_map.items():
            scores = confusions.compute_metrics()
            per_label_scores[label] = scores
            if confusions.TP + confusions.FN > 0:
                precision_sum += scores.precision
                recall_sum += scores.recall
                f1_sum += scores.f1
        num_labels = len(self.label_confusions_map)
        return MacroPRF1Metrics(
            per_label_scores=per_label_scores,
            macro_scores=MacroPRF1Scores(
                num_labels=num_labels,
                precision=safe_division(precision_sum, num_labels),
                recall=safe_division(recall_sum, num_labels),
                f1=safe_division(f1_sum, num_labels),
            ),
        )


class AllConfusions:
    """
    Aggregated class for per label confusions.

    Attributes:
        per_label_confusions: Per label confusion information.
        confusions: Overall TP, FP and FN counts across the labels in
            `per_label_confusions`.
    """

    __slots__ = "per_label_confusions", "confusions"

    def __init__(self) -> None:
        self.per_label_confusions = PerLabelConfusions()
        self.confusions = Confusions()

    def compute_metrics(self) -> PRF1Metrics:
        per_label_metrics = self.per_label_confusions.compute_metrics()
        return PRF1Metrics(
            per_label_scores=per_label_metrics.per_label_scores,
            macro_scores=per_label_metrics.macro_scores,
            micro_scores=self.confusions.compute_metrics(),
        )


def safe_division(n: Union[int, float], d: int) -> float:
    return float(n) / d if d else 0.0


def compute_prf1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = safe_division(tp, tp + fp)
    recall = safe_division(tp, tp + fn)
    f1 = safe_division(2 * tp, 2 * tp + fp + fn)
    return (precision, recall, f1)


def average_precision_score(
    y_true_sorted: np.ndarray, y_score_sorted: np.ndarray
) -> float:
    """
    Computes average precision, which summarizes the precision-recall curve as the
    precisions achieved at each threshold weighted by the increase in recall since the
    previous threshold.

    Args:
        y_true_sorted: Numpy array sorted according to decreasing condifence scores
            indicating whether each prediction is correct.
        y_score_sorted Numpy array of confidence scores for the predictions in
            decreasing order.

    Returns:
        Average precision score.
    """
    ap = 0.0
    tp = 0
    threshold = y_score_sorted[0]
    y_score_sorted = np.append(y_score_sorted[1:], np.NAN)
    total_positive = np.sum(y_true_sorted)
    added_positives = 0

    for k, (label, score) in enumerate(zip(y_true_sorted, y_score_sorted)):
        added_positives += label
        if score != threshold:
            threshold = score
            recall_diff = added_positives / total_positive
            tp += added_positives
            added_positives = 0
            p_at_tresh = tp / (k + 1)
            ap += p_at_tresh * recall_diff
    return float(ap)


def sort_by_score(y_true_list: Sequence[bool], y_score_list: Sequence[float]):
    y_true = np.array(y_true_list)
    y_score = np.array(y_score_list)
    sort_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_true = y_true[sort_indices]
    y_score = y_score[sort_indices]
    return y_true, y_score


def recall_at_precision(
    y_true_sorted: np.ndarray, y_score_sorted: np.ndarray, thresholds: Sequence[float]
) -> Dict[float, float]:
    """
    Computes recall at various precision levels

    Args:
        y_true_sorted: Numpy array sorted according to decreasing condifence scores
            indicating whether each prediction is correct.
        y_score_sorted: Numpy array of confidence scores for the predictions in
            decreasing order.
        thresholds: Sequence of floats indicating the requested precision thresholds

    Returns:
        Dictionary of maximum recall at requested precision thresholds.
    """
    y_score_shift = np.append(y_score_sorted[1:], np.nan)
    score_change = (y_score_sorted - y_score_shift) != 0
    cum_sum = np.cumsum(y_true_sorted)
    recall_at_precision_dict = {t: 0.0 for t in thresholds}
    sum_y_true = y_true_sorted.sum()
    if sum_y_true == 0:
        return recall_at_precision_dict
    recall = cum_sum / sum_y_true
    precision = cum_sum / np.array(range(1, len(y_true_sorted) + 1))
    for threshold in thresholds:
        meets_requirements = np.logical_and(precision >= threshold, score_change)
        r = 0.0
        if np.any(meets_requirements):
            r = float(max(np.extract(meets_requirements, recall)))
        recall_at_precision_dict[threshold] = r
    return recall_at_precision_dict


def compute_soft_metrics(
    predictions: Sequence[LabelPrediction],
    label_names: Sequence[str],
    recall_at_precision_thresholds: Sequence[float] = RECALL_AT_PRECISION_THREHOLDS,
) -> Dict[str, SoftClassificationMetrics]:
    """
    Computes soft classification metrics (for now, average precision) given a list of
    label predictions.

    Args:
        predictions: Label predictions, including the confidence score for each label.
        label_names: Indexed label names.
        recall_at_precision_thresholds: precision thresholds at which to calculate
            recall


    Returns:
        Dict from label strings to their corresponding soft metrics.
    """
    soft_metrics = {}
    for i, label_name in enumerate(label_names):
        y_true = []
        y_score = []
        for label_scores, _, expected in predictions:
            y_true.append(expected == i)
            y_score.append(label_scores[i])
        y_true_sorted, y_score_sorted = sort_by_score(y_true, y_score)
        ap = average_precision_score(y_true_sorted, y_score_sorted)
        recall_at_precision_dict = recall_at_precision(
            y_true_sorted, y_score_sorted, recall_at_precision_thresholds
        )
        soft_metrics[label_name] = SoftClassificationMetrics(
            average_precision=ap, recall_at_precision=recall_at_precision_dict
        )
    return soft_metrics


def compute_matthews_correlation_coefficients(
    TP: int, FP: int, FN: int, TN: int
) -> float:
    """
    Computes Matthews correlation coefficient, a way to summarize all four counts (TP,
    FP, FN, TN) in the confusin matrix of binary classification.

    Args:
        TP: Number of true positives.
        FP: Number of false positives.
        FN: Number of false negatives.
        TN: Number of true negatives.

    Returns:
        Matthews correlation coefficient, which is `sqrt((TP + FP) * (TP + FN) *
        (TN + FP) * (TN + FN))`.
    """
    mcc = safe_division(
        (TP * TN) - (FP * FN),
        np.sqrt(float((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))),
    )
    return mcc


def compute_classification_metrics(
    predictions: Sequence[LabelPrediction],
    label_names: Sequence[str],
    average_precisions: bool = True,
    recall_at_precision_thresholds: Sequence[float] = RECALL_AT_PRECISION_THREHOLDS,
) -> ClassificationMetrics:
    """
    A general function that computes classification metrics given a list of label
    predictions.

    Args:
        predictions: Label predictions, including the confidence score for each label.
        label_names: Indexed label names.
        average_precisions: Whether to compute average precisions for labels or not.
            Defaults to True.
        recall_at_precision_thresholds: precision thresholds at which to calculate recall


    Returns:
        ClassificationMetrics which contains various classification metrics.
    """
    num_correct = 0
    per_label_confusions = PerLabelConfusions()
    for _, predicted, expected in predictions:
        predicted_label = label_names[predicted]
        expected_label = label_names[expected]
        if predicted_label == expected_label:
            num_correct += 1
            per_label_confusions.update(expected_label, "TP", 1)
        else:
            per_label_confusions.update(expected_label, "FN", 1)
            per_label_confusions.update(predicted_label, "FP", 1)
    accuracy = safe_division(num_correct, len(predictions))
    macro_prf1_metrics = per_label_confusions.compute_metrics()

    soft_metrics = (
        compute_soft_metrics(predictions, label_names, recall_at_precision_thresholds)
        if average_precisions
        else None
    )

    if len(label_names) == 2:
        confusion_dict = per_label_confusions.label_confusions_map
        # Since MCC is symmetric, it doesn't matter which label is 0 and which is 1
        TP = confusion_dict[label_names[0]].TP
        FP = confusion_dict[label_names[0]].FP
        FN = confusion_dict[label_names[0]].FN
        TN = confusion_dict[label_names[1]].TP
        mcc: Optional[float] = compute_matthews_correlation_coefficients(TP, FP, FN, TN)
    else:
        mcc = None

    return ClassificationMetrics(
        accuracy=accuracy,
        macro_prf1_metrics=macro_prf1_metrics,
        per_label_soft_scores=soft_metrics,
        mcc=mcc,
    )

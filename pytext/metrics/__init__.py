#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import itertools
from collections import defaultdict
from json import dumps as json_dumps
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
from pytext.common.constants import SpecialTokens
from pytext.utils import cuda
from pytext.utils.ascii_table import ascii_table


NAN_LABELS = [SpecialTokens.UNK, SpecialTokens.PAD]
RECALL_AT_PRECISION_THRESHOLDS = [0.2, 0.4, 0.6, 0.8, 0.9]
PRECISION_AT_RECALL_THRESHOLDS = [0.2, 0.4, 0.6, 0.8, 0.9]

"""
Basic metric classes and functions for single-label prediction problems.
Extending to multi-label support
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


class LabelListPrediction(NamedTuple):
    """
    Label list predictions of an example.

    Attributes:
        label_scores: Confidence scores that each label receives.
        predicted_label: List of indices of the predicted label.
        expected_label: List of indices of the true label.
    """

    label_scores: List[float]
    predicted_label: List[int]
    expected_label: List[int]


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
    decision_thresh_at_precision: Dict[float, float]
    precision_at_recall: Dict[float, float]
    decision_thresh_at_recall: Dict[float, float]
    roc_auc: Optional[float]


class MultiLabelSoftClassificationMetrics(NamedTuple):
    """
    Classification scores that are independent of thresholds.
    """

    average_label_precision: Dict[str, float]
    average_overall_precision: float
    average_label_recall: Dict[str, float]
    average_overall_recall: float
    recall_at_precision: Dict[str, Dict[str, Dict[float, float]]]
    decision_thresh_at_precision: Dict[str, Dict[str, Dict[float, float]]]
    precision_at_recall: Dict[str, Dict[str, Dict[float, float]]]
    decision_thresh_at_recall: Dict[str, Dict[str, Dict[float, float]]]
    roc_auc: Optional[Dict[Optional[str], Optional[Dict[str, Optional[float]]]]]
    average_overall_auc: float
    label_accuracy: Dict[str, float]
    average_overall_accuracy: float


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

    def print_metrics(self, indentation="") -> None:
        print(
            ascii_table(
                [
                    {
                        "label": label,
                        "precision": f"{metrics.precision:.2f}",
                        "recall": f"{metrics.recall:.2f}",
                        "f1": f"{metrics.f1:.2f}",
                        "support": metrics.true_positives + metrics.false_negatives,
                    }
                    for label, metrics in sorted(self.per_label_scores.items())
                ],
                human_column_names={
                    "label": "Label",
                    "precision": "Precision",
                    "recall": "Recall",
                    "f1": "F1",
                    "support": "Support",
                },
                footer={
                    "label": "Overall macro scores",
                    "precision": f"{self.macro_scores.precision:.2f}",
                    "recall": f"{self.macro_scores.recall:.2f}",
                    "f1": f"{self.macro_scores.f1:.2f}",
                },
                alignments={"label": "<"},
                indentation=indentation,
            )
        )


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
                f"\t{label_metrics.precision * 100:<10.3f}"
                f"\t{label_metrics.recall * 100:<10.3f}"
                f"\t{label_metrics.f1 * 100:<10.3f}"
                f"\t{support:<10}\n"
            )
        support = self.micro_scores.true_positives + self.micro_scores.false_negatives
        res += (
            f"\n\t{'Overall micro scores':<40}"
            f"\t{self.micro_scores.precision * 100:<10.3f}"
            f"\t{self.micro_scores.recall * 100:<10.3f}"
            f"\t{self.micro_scores.f1 * 100:<10.3f}"
            f"\t{support:<10}\n"
        )
        res += (
            f"\t{'Overall macro scores':<40}"
            f"\t{self.macro_scores.precision * 100:<10.3f}"
            f"\t{self.macro_scores.recall * 100:<10.3f}"
            f"\t{self.macro_scores.f1 * 100:<10.3f}\n"
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
        roc_auc: Area under the Receiver Operating Characteristic curve.
        loss: Training loss (only used for selecting best model, no need to print).
    """

    accuracy: float
    macro_prf1_metrics: MacroPRF1Metrics
    per_label_soft_scores: Optional[Dict[str, SoftClassificationMetrics]]
    mcc: Optional[float]
    roc_auc: Optional[float]
    loss: float

    def print_metrics(self, report_pep=False) -> None:
        print(f"Accuracy: {self.accuracy * 100:.2f}")
        print("\nSoft Metrics:")
        if self.per_label_soft_scores:
            soft_scores = [
                {
                    "label": label,
                    "avg_pr": f"{metrics.average_precision:.3f}",
                    "roc_auc": f"{(metrics.roc_auc or 0.0):.3f}",
                }
                for label, metrics in sorted(self.per_label_soft_scores.items())
            ]
            columns = {
                "label": "Label",
                "avg_pr": "Average precision",
                "roc_auc": "ROC AUC",
            }
            print(ascii_table(soft_scores, columns))
            print("\nRecall at Precision")
            r_at_p_thresholds = set(
                itertools.chain.from_iterable(
                    metrics.recall_at_precision
                    for metrics in self.per_label_soft_scores.values()
                )
            )
            print(
                ascii_table(
                    (
                        dict(
                            {"label": label},
                            **{
                                str(p): f"{r:.3f}"
                                for p, r in metrics.recall_at_precision.items()
                            },
                        )
                        for label, metrics in sorted(self.per_label_soft_scores.items())
                    ),
                    dict(
                        {"label": "Label"},
                        **{str(t): f"R@P {t}" for t in r_at_p_thresholds},
                    ),
                    alignments={"label": "<"},
                )
            )
            print("\nPrecision at Recall")
            p_at_r_thresholds = set(
                itertools.chain.from_iterable(
                    metrics.precision_at_recall
                    for metrics in self.per_label_soft_scores.values()
                )
            )
            print(
                ascii_table(
                    (
                        dict(
                            {"label": label},
                            **{
                                str(p): f"{r:.3f}"
                                for p, r in metrics.precision_at_recall.items()
                            },
                        )
                        for label, metrics in sorted(self.per_label_soft_scores.items())
                    ),
                    dict(
                        {"label": "Label"},
                        **{str(t): f"P@R {t}" for t in p_at_r_thresholds},
                    ),
                    alignments={"label": "<"},
                )
            )
        if self.mcc:
            print(f"\nMatthews correlation coefficient: {self.mcc :.3f}")
        if self.roc_auc:
            print(f"\nROC AUC: {self.roc_auc:.3f}")
        if report_pep:
            self.print_pep()

    def print_pep(self):
        metrics = {"Accuracy": f"{self.accuracy * 100:.2f}"}
        if self.roc_auc:
            metrics["ROC AUC"] = f"{self.roc_auc :.3f}"
        for key, value in metrics.items():
            info = {"type": "NET", "metric": key, "unit": "None", "value": value}
            print("PyTorchObserver " + json_dumps(info))


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
        for label, confusions in sorted(self.label_confusions_map.items()):
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


class PairwiseRankingMetrics(NamedTuple):
    """
    Metric class for pairwise ranking

    Attributes:
        num_examples (int): number of samples
        accuracy (float): how many times did we rank in the correct order
        average_score_difference (float): average score(higherRank) - score(lowerRank)

    """

    num_examples: int
    accuracy: float
    average_score_difference: float

    def print_metrics(self) -> None:
        print(f"RankingAccuracy: {self.accuracy * 100:.2f}")
        print(f"AvgScoreDiff: {self.average_score_difference}")
        print(f"NumExamples: {self.num_examples}")


class RegressionMetrics(NamedTuple):
    """
    Metrics for regression tasks.

    Attributes:
        num_examples (int): number of examples
        pearson_correlation (float): correlation between predictions and labels
        mse (float): mean-squared error between predictions and labels
    """

    num_examples: int
    pearson_correlation: float
    mse: float

    def print_metrics(self):
        print(f"Num examples: {self.num_examples}")
        print(f"Pearson correlation: {self.pearson_correlation:.3f}")
        print(f"Mean squared error: {self.mse:.3f}")


class RealtimeMetrics(NamedTuple):
    """
    Realtime Metrics for tracking training progress and performance.

    Attributes:
        samples (int): number of samples
        tps (float): tokens per second
        ups (float): updates per second
    """

    samples: int
    tps: float
    ups: float

    def _format(self, key, value):
        if key in ("tps", "ups"):
            return round(value)
        return value

    def __str__(self):
        metrics = {"num_gpus": cuda.DISTRIBUTED_WORLD_SIZE}
        for key, value in self._asdict().items():
            if not value:
                continue
            metrics[key] = self._format(key, value)
        return str(metrics)


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
        y_true_sorted: Numpy array sorted according to decreasing confidence scores
            indicating whether each prediction is correct.
        y_score_sorted Numpy array of confidence scores for the predictions in
            decreasing order.

    Returns:
        Average precision score.

    TODO: This is too slow, improve the performance
    """
    ap = 0.0
    tp = 0
    threshold = y_score_sorted[0]
    y_score_sorted = np.append(y_score_sorted[1:], np.NAN)
    total_positive = np.sum(y_true_sorted)
    added_positives = 0

    for k, (label, score) in enumerate(zip(y_true_sorted, y_score_sorted)):
        if label:
            added_positives += 1
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
        y_true_sorted: Numpy array sorted according to decreasing confidence scores
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
    decision_thresh_at_precision_dict = {t: 0.0 for t in thresholds}
    sum_y_true = y_true_sorted.sum()
    if sum_y_true == 0:
        return recall_at_precision_dict, decision_thresh_at_precision_dict
    recall = cum_sum / sum_y_true
    precision = cum_sum / np.array(range(1, len(y_true_sorted) + 1))

    for threshold in thresholds:
        meets_requirements = np.logical_and(precision >= threshold, score_change)
        if not np.any(meets_requirements):
            continue

        recall_at_precision_dict[threshold] = float(
            max(np.extract(meets_requirements, recall))
        )
        decision_thresh_at_precision_dict[threshold] = float(
            min(np.extract(meets_requirements, y_score_sorted))
        )

    return recall_at_precision_dict, decision_thresh_at_precision_dict


def precision_at_recall(
    y_true_sorted: np.ndarray, y_score_sorted: np.ndarray, thresholds: Sequence[float]
) -> Tuple[Dict[float, float], Dict[float, float]]:
    """
    Computes precision at various recall levels

    Args:
        y_true_sorted: Numpy array sorted according to decreasing confidence scores
            indicating whether each prediction is correct.
        y_score_sorted: Numpy array of confidence scores for the predictions in
            decreasing order.
        thresholds: Sequence of floats indicating the requested recall thresholds

    Returns:
        Dictionary of maximum precision at requested recall thresholds.
        Dictionary of decision thresholds resulting in max precision at
        requested recall thresholds.
    """
    y_score_shift = np.append(y_score_sorted[1:], np.nan)
    score_change = (y_score_sorted - y_score_shift) != 0
    cum_sum = np.cumsum(y_true_sorted)
    precision_at_recall_dict = {t: 0.0 for t in thresholds}
    decision_thresh_at_recall_dict = {t: 0.0 for t in thresholds}
    sum_y_true = y_true_sorted.sum()
    if sum_y_true == 0:
        return precision_at_recall_dict, decision_thresh_at_recall_dict
    recall = cum_sum / sum_y_true
    precision = cum_sum / np.array(range(1, len(y_true_sorted) + 1))
    for threshold in thresholds:
        meets_requirements = np.logical_and(recall >= threshold, score_change)
        if not np.any(meets_requirements):
            continue
        precisions_meeting_requirements = np.extract(meets_requirements, precision)
        idx_max_precision_at_recall = np.amin(
            np.argmax(precisions_meeting_requirements), axis=None
        )
        precision_at_recall_dict[threshold] = float(
            precisions_meeting_requirements[idx_max_precision_at_recall]
        )
        decision_thresh_at_recall_dict[threshold] = float(
            np.extract(meets_requirements, y_score_sorted)[idx_max_precision_at_recall]
        )
    return precision_at_recall_dict, decision_thresh_at_recall_dict


def compute_average_recall(
    predictions: Sequence[LabelPrediction],
    label_names: Sequence[str],
    average_precisions: Dict[str, float],
) -> float:
    recalls = []
    for i, label_name in enumerate(label_names):
        y_true = []
        y_score = []
        for label_scores, _, expected in predictions:
            y_true.append(expected == i)
            y_score.append(label_scores[i])
        y_true_sorted, y_score_sorted = sort_by_score(y_true, y_score)
        recall_at_precision_dict, _ = recall_at_precision(
            y_true_sorted, y_score_sorted, [average_precisions[label_name]]
        )
        for _, value in recall_at_precision_dict.items():
            recalls.append(value)
    return sum(v for v in recalls) / (len(recalls) * 1.0)


def compute_soft_metrics(
    predictions: Sequence[LabelPrediction],
    label_names: Sequence[str],
    recall_at_precision_thresholds: Sequence[float] = RECALL_AT_PRECISION_THRESHOLDS,
    precision_at_recall_thresholds: Sequence[float] = PRECISION_AT_RECALL_THRESHOLDS,
) -> Dict[str, SoftClassificationMetrics]:
    """
    Computes soft classification metrics given a list of label predictions.

    Args:
        predictions: Label predictions, including the confidence score for each label.
        label_names: Indexed label names.
        recall_at_precision_thresholds: precision thresholds at which to calculate
            recall
        precision_at_recall_thresholds: recall thresholds at which to calculate
            precision


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
        recall_at_precision_dict, decision_thresh_at_precision = recall_at_precision(
            y_true_sorted, y_score_sorted, recall_at_precision_thresholds
        )
        precision_at_recall_dict, decision_thresh_at_recall = precision_at_recall(
            y_true_sorted, y_score_sorted, precision_at_recall_thresholds
        )
        roc_auc = compute_roc_auc(predictions, target_class=i)
        soft_metrics[label_name] = SoftClassificationMetrics(
            average_precision=ap,
            recall_at_precision=recall_at_precision_dict,
            decision_thresh_at_precision=decision_thresh_at_precision,
            precision_at_recall=precision_at_recall_dict,
            decision_thresh_at_recall=decision_thresh_at_recall,
            roc_auc=roc_auc,
        )
    return soft_metrics


def compute_multi_label_soft_metrics(
    predictions: Sequence[LabelListPrediction],
    label_names: Sequence[str],
    recall_at_precision_thresholds: Sequence[float] = RECALL_AT_PRECISION_THRESHOLDS,
    precision_at_recall_thresholds: Sequence[float] = PRECISION_AT_RECALL_THRESHOLDS,
) -> Dict[str, SoftClassificationMetrics]:
    """
    Computes multi-label soft classification metrics

    Args:
        predictions: multi-label predictions,
                     including the confidence score for each label.
        label_names: Indexed label names.
        recall_at_precision_thresholds: precision thresholds at which to calculate
            recall
        precision_at_recall_thresholds: recall thresholds at which to calculate
            precision


    Returns:
        Dict from label strings to their corresponding soft metrics.
    """
    soft_metrics = {}
    for i, label_name in enumerate(label_names):
        y_true = []
        y_score = []
        for label_scores, _, expected in predictions:
            y_true.append(i in expected)
            y_score.append(label_scores[i])
        y_true_sorted, y_score_sorted = sort_by_score(y_true, y_score)
        ap = average_precision_score(y_true_sorted, y_score_sorted)
        recall_at_precision_dict, decision_thresh_at_precision = recall_at_precision(
            y_true_sorted, y_score_sorted, recall_at_precision_thresholds
        )
        precision_at_recall_dict, decision_thresh_at_recall = precision_at_recall(
            y_true_sorted, y_score_sorted, precision_at_recall_thresholds
        )
        roc_auc = compute_roc_auc(predictions, target_class=i)
        soft_metrics[label_name] = SoftClassificationMetrics(
            average_precision=ap,
            recall_at_precision=recall_at_precision_dict,
            decision_thresh_at_precision=decision_thresh_at_precision,
            precision_at_recall=precision_at_recall_dict,
            decision_thresh_at_recall=decision_thresh_at_recall,
            roc_auc=roc_auc,
        )
    return soft_metrics


def compute_multi_label_multi_class_soft_metrics(
    predictions: Sequence[Sequence[LabelPrediction]],
    label_names: Sequence[str],
    label_vocabs: Sequence[Sequence[str]],
    recall_at_precision_thresholds: Sequence[float] = RECALL_AT_PRECISION_THRESHOLDS,
    precision_at_recall_thresholds: Sequence[float] = PRECISION_AT_RECALL_THRESHOLDS,
) -> MultiLabelSoftClassificationMetrics:
    """

    Computes multi-label soft classification metrics with multi-class accommodation

    Args:
        predictions: multi-label predictions,
                     including the confidence score for each label.
        label_names: Indexed label names.
        recall_at_precision_thresholds: precision thresholds at which to calculate
            recall
        precision_at_recall_thresholds: recall thresholds at which to calculate
            precision


    Returns:
        Dict from label strings to their corresponding soft metrics.
    """

    average_precision = {}
    average_recall = {}
    recall_at_precision = {}
    decision_thresh_at_precision = {}
    precision_at_recall = {}
    decision_thresh_at_recall = {}
    roc_auc = {}
    class_accuracy = {}
    average_auc = []

    for label_idx, label_vocab in enumerate(label_vocabs):
        label = list(label_names)[label_idx]
        avg = (
            sum(1 for s, p, e in predictions[label_idx] if p == e)
            / len(predictions[label_idx])
            * 1.0
        )
        class_accuracy[label] = avg
        soft_metrics_ = compute_soft_metrics(predictions[label_idx], label_vocab)
        temp_avg_precision_ = {k: v.average_precision for k, v in soft_metrics_.items()}
        average_precision[label] = sum(
            v for k, v in temp_avg_precision_.items() if k not in NAN_LABELS
        ) / (
            sum(1 for k, v in temp_avg_precision_.items() if k not in NAN_LABELS) * 1.0
        )

        average_recall[label] = compute_average_recall(
            predictions[label_idx], label_vocab, temp_avg_precision_
        )
        recall_at_precision[label] = {
            k: v.recall_at_precision for k, v in soft_metrics_.items()
        }
        decision_thresh_at_precision[label] = {
            k: v.decision_thresh_at_precision for k, v in soft_metrics_.items()
        }
        precision_at_recall[label] = {
            k: v.precision_at_recall for k, v in soft_metrics_.items()
        }
        decision_thresh_at_recall[label] = {
            k: v.decision_thresh_at_recall for k, v in soft_metrics_.items()
        }
        roc_auc[label] = {k: v.roc_auc for k, v in soft_metrics_.items()}
        average_auc.append(
            sum(v for v in roc_auc[label].values()) / (len(roc_auc[label]) * 1.0)
        )

    return MultiLabelSoftClassificationMetrics(
        average_label_precision=average_precision,
        average_overall_precision=sum(v for v in average_precision.values())
        / (len(average_precision) * 1.0),
        average_label_recall=average_recall,
        average_overall_recall=sum(v for v in average_recall.values())
        / (len(average_recall) * 1.0),
        recall_at_precision=recall_at_precision,
        decision_thresh_at_precision=decision_thresh_at_precision,
        precision_at_recall=precision_at_recall,
        decision_thresh_at_recall=decision_thresh_at_recall,
        roc_auc=roc_auc,
        average_overall_auc=sum(v for v in average_auc) / (len(average_auc) * 1.0),
        label_accuracy=class_accuracy,
        average_overall_accuracy=sum(v for v in class_accuracy.values())
        / (len(class_accuracy) * 1.0),
    )


def compute_matthews_correlation_coefficients(
    TP: int, FP: int, FN: int, TN: int
) -> float:
    """
    Computes Matthews correlation coefficient, a way to summarize all four counts (TP,
    FP, FN, TN) in the confusion matrix of binary classification.

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


def compute_roc_auc(
    predictions: Sequence[LabelPrediction], target_class: int = 0
) -> Optional[float]:
    """
    Computes area under the Receiver Operating Characteristic curve, for binary
    classification. Implementation based off of (and explained at)
    https://www.ibm.com/developerworks/community/blogs/jfp/entry/Fast_Computation_of_AUC_ROC_score?lang=en.
    """
    # Collect scores
    y_true = [expected == target_class for _, _, expected in predictions]
    y_score = [label_scores[target_class] for label_scores, _, _ in predictions]
    y_true_sorted, _ = sort_by_score(y_true, y_score)

    # Compute auc as probability that a positive example is scored higher than
    # a negative example.
    n_false = 0
    n_correct_pair_order = 0

    for y in reversed(y_true_sorted):  # want low predicted to high predicted
        if y:
            n_correct_pair_order += n_false
        else:
            n_false += 1

    n_true = len(y_true) - n_false
    if n_true == 0 or n_false == 0:
        return None

    return float(n_correct_pair_order / (n_true * n_false))


def compute_classification_metrics(
    predictions: Sequence[LabelPrediction],
    label_names: Sequence[str],
    loss: float,
    average_precisions: bool = True,
    recall_at_precision_thresholds: Sequence[float] = RECALL_AT_PRECISION_THRESHOLDS,
    precision_at_recall_thresholds: Sequence[float] = PRECISION_AT_RECALL_THRESHOLDS,
) -> ClassificationMetrics:
    """
    A general function that computes classification metrics given a list of label
    predictions.

    Args:
        predictions: Label predictions, including the confidence score for each label.
        label_names: Indexed label names.
        average_precisions: Whether to compute average precisions for labels or not.
            Defaults to True.
        recall_at_precision_thresholds: precision thresholds at which
                                        to calculate recall
        precision_at_recall_thresholds: recall thresholds at which
                                        to calculate precision


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
        compute_soft_metrics(
            predictions,
            label_names,
            recall_at_precision_thresholds,
            precision_at_recall_thresholds,
        )
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
        roc_auc: Optional[float] = compute_roc_auc(predictions)
    else:
        mcc = None
        roc_auc = None

    return ClassificationMetrics(
        accuracy=accuracy,
        macro_prf1_metrics=macro_prf1_metrics,
        per_label_soft_scores=soft_metrics,
        mcc=mcc,
        roc_auc=roc_auc,
        loss=loss,
    )


def compute_multi_label_classification_metrics(
    predictions: Sequence[LabelListPrediction],
    label_names: Sequence[str],
    loss: float,
    average_precisions: bool = True,
    recall_at_precision_thresholds: Sequence[float] = RECALL_AT_PRECISION_THRESHOLDS,
    precision_at_recall_thresholds: Sequence[float] = PRECISION_AT_RECALL_THRESHOLDS,
) -> ClassificationMetrics:
    """
    A general function that computes classification metrics given a list of multi-label
    predictions.

    Args:
        predictions: multi-label predictions,
                     including the confidence score for each label.
        label_names: Indexed label names.
        average_precisions: Whether to compute average precisions for labels or not.
                            Defaults to True.
        recall_at_precision_thresholds: precision thresholds at which
                                        to calculate recall
        precision_at_recall_thresholds: recall thresholds at which
                                        to calculate precision


    Returns:
        ClassificationMetrics which contains various classification metrics.
    """

    num_correct = 0
    num_expected_labels = 0
    per_label_confusions = PerLabelConfusions()
    for _, predicted, expected in predictions:
        for label_idx, label_name in enumerate(label_names):
            num_expected_labels += 1
            # "predicted" is in the format of n_hot_encoding
            if predicted[label_idx] == 1:
                if label_idx in expected:  # TP
                    num_correct += 1
                    per_label_confusions.update(label_name, "TP", 1)
                else:  # FP
                    per_label_confusions.update(label_name, "FP", 1)
            else:
                if label_idx in expected:  # FN
                    per_label_confusions.update(label_name, "FN", 1)
                else:  # TN, update correct num
                    num_correct += 1

    accuracy = safe_division(num_correct, num_expected_labels)
    macro_prf1_metrics = per_label_confusions.compute_metrics()

    soft_metrics = (
        compute_multi_label_soft_metrics(
            predictions,
            label_names,
            recall_at_precision_thresholds,
            precision_at_recall_thresholds,
        )
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
        roc_auc: Optional[float] = compute_roc_auc(predictions)
    else:
        mcc = None
        roc_auc = None

    return ClassificationMetrics(
        accuracy=accuracy,
        macro_prf1_metrics=macro_prf1_metrics,
        per_label_soft_scores=soft_metrics,
        mcc=mcc,
        roc_auc=roc_auc,
        loss=loss,
    )


def compute_pairwise_ranking_metrics(
    predictions: Sequence[int], scores: Sequence[float]
) -> PairwiseRankingMetrics:
    """
    Computes metrics for pairwise ranking given sequences of predictions and scores

    Args:
        predictions : 1 if ranking was correct, 0 if ranking was incorrect
        scores : score(higher-ranked-sample) - score(lower-ranked-sample)

    Returns:
        PairwiseRankingMetrics object
    """
    return PairwiseRankingMetrics(
        num_examples=len(predictions),
        accuracy=safe_division(sum(predictions), len(predictions)),
        average_score_difference=safe_division(sum(scores), len(predictions)),
    )


def compute_regression_metrics(
    predictions: Sequence[float], targets: Sequence[float]
) -> RegressionMetrics:
    """
    Computes metrics for regression tasks.abs

    Args:
        predictions: 1-D sequence of float predictions
        targets: 1-D sequence of float labels

    Returns:
        RegressionMetrics object
    """
    preds, targs = np.array(predictions), np.array(targets)
    pred_mean, targ_mean = preds.mean(), targs.mean()
    covariance = (preds - pred_mean).dot(targs - targ_mean) / preds.size
    corr = covariance / preds.std() / targs.std()

    mse = np.square(preds - targs).mean()
    return RegressionMetrics(num_examples=len(preds), pearson_correlation=corr, mse=mse)

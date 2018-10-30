#!/usr/bin/env python3

from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np


class LabelPrediction(NamedTuple):
    label_scores: List[float]
    predicted_label: int
    expected_label: int


class PRF1Scores(NamedTuple):
    """
    Class for the typical precision/recall/F1 scores where
      precision = TP / (TP + FP)
      recall = TP / (TP + FN)
      f1 = 2 * TP / (2 * TP + FP + FN)
    This class is used for both micro scores and per label scores.
    """

    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float


class SoftClassificationMetrics(NamedTuple):
    """
    Class for non-threshold dependent classification score
    """

    average_precision: float


class MacroPRF1Scores(NamedTuple):
    """
    The macro precision/recall/F1 scores which are averages across the per label
    scores.
    num_label: stores how many labels are used in the average.
    """

    num_labels: int
    precision: float
    recall: float
    f1: float


class MacroPRF1Metrics(NamedTuple):
    """
    Return type of PerLabelConfusions.compute_metrics(). It contains both
    per_label_scores and macro_scores because they are both computed on per label
    basis.
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
    All kinds of precision/recall/F1 scores that we want to report.
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
    Class of various classification metrics, including precision/recall/F1 scores,
    Matthews correlation coefficient and average precision.
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
            print(f"\t{'Label':<10}{'Average precision':<10}")
            for label, label_metrics in self.per_label_soft_scores.items():
                print(f"\t{label:<10}{label_metrics.average_precision * 100:<10.2f}")
        if self.mcc:
            print(f"\nMatthews correlation coefficient: {self.mcc :.2f}")


# We need a mutable type here (instead of NamedTuple) because we want to
# aggregate TP, FP and FN
class Confusions:
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
    __slots__ = "label_confusions_map"

    def __init__(self) -> None:
        self.label_confusions_map: DefaultDict[str, Confusions] = defaultdict(
            Confusions
        )

    def update(self, label: str, item: str, count: int) -> None:
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
    A class that stores per label confusions as well as the total confusion counts
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


def compute_classification_metrics(
    predictions: List[LabelPrediction],
    label_names: List[str],
    average_precisions: bool = True,
) -> ClassificationMetrics:
    """
    A general function that computes classification metrics given a list of pairs of
    predicted and expected labels.
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
        compute_soft_metrics(predictions, label_names) if average_precisions else None
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


def compute_soft_metrics(
    predictions: List[LabelPrediction], label_names: List[str]
) -> Dict[str, SoftClassificationMetrics]:
    """
    Computes classification metrics given a list of pairs of scores and expected labels.
    """

    soft_metrics = {}
    for i, label_name in enumerate(label_names):
        y_true = []
        y_score = []
        for label_scores, _, expected in predictions:
            y_true.append(expected == i)
            y_score.append(label_scores[i])
        ap = average_precision_score(y_true, y_score)
        soft_metrics[label_name] = SoftClassificationMetrics(average_precision=ap)
    return soft_metrics


def compute_matthews_correlation_coefficients(
    TP: int, FP: int, FN: int, TN: int
) -> float:
    """
    Matthews correlation coefficient is a way to summarize all four values of
    the confusin matrix for binary classification.
    """

    mcc = safe_division(
        (TP * TN) - (FP * FN),
        np.sqrt(float((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))),
    )
    return mcc


def average_precision_score(
    y_true_list: List[bool], y_score_list: List[float]
) -> float:
    """
    summarizes the precision-recall curve, as the precisions achieved at each
    threshold, weighted by the increase in recall since previous threshold
    """

    y_true = np.array(y_true_list)
    y_score = np.array(y_score_list)
    sort_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_true = y_true[sort_indices]
    y_score = y_score[sort_indices]
    ap = 0.0
    tp = 0
    threshold = y_score[0]
    y_score = np.append(y_score[1:], np.NAN)
    total_positive = np.sum(y_true)
    added_positives = 0

    for k, (label, score) in enumerate(zip(y_true, y_score)):
        added_positives += label
        if score != threshold:
            threshold = score
            recall_diff = added_positives / total_positive
            tp += added_positives
            added_positives = 0
            p_at_tresh = tp / (k + 1)
            ap += p_at_tresh * recall_diff
    return float(ap)

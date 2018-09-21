#!/usr/bin/env python3
import logging
from collections import Counter as counter, defaultdict
from copy import deepcopy
from typing import (
    Any,
    Counter,
    DefaultDict,
    Dict,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Union,
)

import numpy as np


logger = logging.getLogger(name=__name__)


class Span(NamedTuple):
    """
    Span of a node in an utterance. Absolute positions are used here.
    """

    start: int
    end: int


class Node:
    """
    A class that represents nodes in a parse tree, used for identifying the true
    positives in both tree-based and bracketing metrics.
    """

    __slots__ = "label", "span", "children"

    def __init__(self, label: str, span: Span, children: Set["Node"] = None) -> None:
        self.label: str = label
        self.span: Span = span
        # This will be left empty when computing bracketing metrics.
        self.children: Set[Node] = children or set()

    def __hash__(self):
        return hash((self.label, self.span, frozenset(self.children)))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Node):
            return NotImplemented
        return (
            self.label == other.label
            and self.span == other.span
            and self.children == other.children
        )

    def get_depth(self) -> int:
        return 1 + max((child.get_depth() for child in self.children), default=0)


class LabelPredictionPair(NamedTuple):
    predicted_label: str
    expected_label: str


class LabelScoresPair(NamedTuple):
    predicted_scores: List[float]
    expected_label: str


class FramePredictionPair(NamedTuple):
    predicted_frame: Node
    expected_frame: Node


class NodesPredictionPair(NamedTuple):
    predicted_nodes: Counter[Node]
    expected_nodes: Counter[Node]


class IntentsAndSlots(NamedTuple):
    intents: Counter[Node]
    slots: Counter[Node]


class FrameAccuracy(NamedTuple):
    num_samples: int
    frame_accuracy: float


FrameAccuraciesByDepth = Dict[int, FrameAccuracy]


class ClassificationMetrics(NamedTuple):
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


class MacroClassificationMetrics(NamedTuple):
    """
    The macro precision/recall/F1 scores which are averages across the per label
    scores.
    num_label: stores how many labels are used in the average.
    """

    num_labels: int
    precision: float
    recall: float
    f1: float


class PerLabelMetrics(NamedTuple):
    """
    Return type of compute_metrics() in class PerLabelMetrics. It contains both
    per_label_scores and macro_scores because they are both computed on per label
    basis.
    """

    per_label_scores: Dict[str, ClassificationMetrics]
    macro_scores: MacroClassificationMetrics


class AllClassificationMetrics(NamedTuple):
    """
    All kinds of precision/recall/F1 scores that we want to report.
    """

    per_label_scores: Dict[str, ClassificationMetrics]
    macro_scores: MacroClassificationMetrics
    micro_scores: ClassificationMetrics

    def print_metrics(self) -> None:
        print("Per label scores:")
        for label, label_metrics in self.per_label_scores.items():
            print(
                f"  Label: {label}, P = {label_metrics.precision * 100:.2f}, "
                f"R = {label_metrics.recall * 100:.2f}, "
                f"F1 = {label_metrics.f1 * 100:.2f};"
            )
        print("Overall micro scores:")
        print(
            f"  P = {self.micro_scores.precision * 100:.2f} "
            f"R = {self.micro_scores.recall * 100:.2f}, "
            f"F1 = {self.micro_scores.f1 * 100:.2f}."
        )
        print("Overall macro scores:")
        print(
            f"  P = {self.macro_scores.precision * 100:.2f} "
            f"R = {self.macro_scores.recall * 100:.2f}, "
            f"F1 = {self.macro_scores.f1 * 100:.2f}."
        )


class BinaryClassificationMetrics(NamedTuple):
    """
    Binary classification metrics such as MCC and average precision.
    """

    all_classification_metrics: AllClassificationMetrics
    per_label_soft_scores: Dict[str, SoftClassificationMetrics]
    mcc: float

    def print_metrics(self) -> None:
        self.all_classification_metrics.print_metrics()
        print("Per label soft scores:")
        for label, label_metrics in self.per_label_soft_scores.items():
            print(
                f"  Label: {label}, AP = {label_metrics.average_precision * 100:.2f}, "
            )
        print(f" MCC = {self.mcc :.2f}")


class IntentSlotMetrics:
    """
    Intent and slot metrics.
    """

    __slots__ = "intent_metrics", "slot_metrics", "overall_metrics"

    def __init__(
        self,
        intent_metrics: Optional[AllClassificationMetrics],
        slot_metrics: Optional[AllClassificationMetrics],
        overall_metrics: Optional[ClassificationMetrics],
    ) -> None:
        self.intent_metrics = intent_metrics
        self.slot_metrics = slot_metrics
        self.overall_metrics = overall_metrics

    def print_metrics(self) -> None:
        if self.intent_metrics:
            print("\nIntent Metrics")
            self.intent_metrics.print_metrics()
        if self.slot_metrics:
            print("\nSlot Metrics")
            self.slot_metrics.print_metrics()
        if self.overall_metrics:
            print("\nMerged Intent and Slot Metrics")
            print(
                f"  P = {self.overall_metrics.precision * 100:.2f} "
                f"R = {self.overall_metrics.recall * 100:.2f}, "
                f"F1 = {self.overall_metrics.f1 * 100:.2f}."
            )


class AllMetrics:
    __slots__ = (
        "top_intent_accuracy",
        "frame_accuracy",
        "frame_accuracies_by_depth",
        "bracket_metrics",
        "tree_metrics",
    )

    def __init__(
        self,
        top_intent_accuracy: Optional[float],
        frame_accuracy: Optional[float],
        frame_accuracies_by_depth: Optional[FrameAccuraciesByDepth],
        bracket_metrics: Optional[IntentSlotMetrics],
        tree_metrics: Optional[IntentSlotMetrics],
    ) -> None:
        self.top_intent_accuracy = top_intent_accuracy
        self.frame_accuracy = frame_accuracy
        self.frame_accuracies_by_depth = frame_accuracies_by_depth
        self.bracket_metrics = bracket_metrics
        self.tree_metrics = tree_metrics

    def print_metrics(self) -> None:
        if self.frame_accuracy:
            print(f"\n\nFrame accuracy = {self.frame_accuracy * 100:.2f}")
        if self.bracket_metrics:
            print("\n\nBracket Metrics")
            self.bracket_metrics.print_metrics()
        if self.tree_metrics:
            print("\n\nTree Metrics")
            self.tree_metrics.print_metrics()


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

    def compute_metrics(self) -> ClassificationMetrics:
        precision, recall, f1 = compute_prf1(self.TP, self.FP, self.FN)
        return ClassificationMetrics(
            true_positives=self.TP,
            false_positives=self.FP,
            false_negatives=self.FN,
            precision=precision,
            recall=recall,
            f1=f1,
        )


class IntentSlotConfusions(NamedTuple):
    intent_confusions: Confusions
    slot_confusions: Confusions


class PerLabelConfusions:
    __slots__ = "dictionary"

    def __init__(self) -> None:
        self.dictionary: DefaultDict[str, Confusions] = defaultdict(Confusions)

    def update(self, label: str, item: str, count: int) -> None:
        confusions = self.dictionary[label]
        setattr(confusions, item, getattr(confusions, item) + count)

    def compute_metrics(self) -> PerLabelMetrics:
        per_label_scores: Dict[str, ClassificationMetrics] = {}
        precision_sum, recall_sum, f1_sum = 0.0, 0.0, 0.0
        for label, confusions in self.dictionary.items():
            scores = confusions.compute_metrics()
            per_label_scores[label] = scores
            if confusions.TP + confusions.FN > 0:
                precision_sum += scores.precision
                recall_sum += scores.recall
                f1_sum += scores.f1
        num_labels = len(self.dictionary)
        return PerLabelMetrics(
            per_label_scores=per_label_scores,
            macro_scores=MacroClassificationMetrics(
                num_labels=num_labels,
                precision=_safe_division(precision_sum, num_labels),
                recall=_safe_division(recall_sum, num_labels),
                f1=_safe_division(f1_sum, num_labels),
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

    def compute_metrics(self) -> AllClassificationMetrics:
        per_label_metrics = self.per_label_confusions.compute_metrics()
        return AllClassificationMetrics(
            per_label_scores=per_label_metrics.per_label_scores,
            macro_scores=per_label_metrics.macro_scores,
            micro_scores=self.confusions.compute_metrics(),
        )


def _safe_division(n: Union[int, float], d: int) -> float:
    return float(n) / d if d else 0.0


def compute_prf1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = _safe_division(tp, tp + fp)
    recall = _safe_division(tp, tp + fn)
    f1 = _safe_division(2 * tp, 2 * tp + fp + fn)
    return (precision, recall, f1)


def _compare_nodes(
    predicted_nodes: Counter[Node],
    expected_nodes: Counter[Node],
    per_label_confusions: Optional[PerLabelConfusions] = None,
) -> Confusions:
    true_positives = predicted_nodes & expected_nodes
    false_positives = predicted_nodes - true_positives
    false_negatives = expected_nodes - true_positives

    if per_label_confusions:
        for node, count in true_positives.items():
            per_label_confusions.update(node.label, "TP", count)
        for node, count in false_positives.items():
            per_label_confusions.update(node.label, "FP", count)
        for node, count in false_negatives.items():
            per_label_confusions.update(node.label, "FN", count)

    return Confusions(
        TP=sum(true_positives.values()),
        FP=sum(false_positives.values()),
        FN=sum(false_negatives.values()),
    )


def _get_intents_and_slots(frame: Node, tree_based: bool) -> IntentsAndSlots:
    intents: Counter[Node] = counter()
    slots: Counter[Node] = counter()

    def process_node(node: Node, is_intent: bool) -> None:
        for child in node.children:
            process_node(child, not is_intent)
        if not tree_based:
            node = Node(node.label, deepcopy(node.span))
        if is_intent:
            intents[node] += 1
        else:
            slots[node] += 1

    process_node(frame, True)
    return IntentsAndSlots(intents=intents, slots=slots)


def compare_frames(
    predicted_frame: Node,
    expected_frame: Node,
    tree_based: bool,
    intent_per_label_confusions: Optional[PerLabelConfusions] = None,
    slot_per_label_confusions: Optional[PerLabelConfusions] = None,
) -> IntentSlotConfusions:
    """
    Compare two frames of "Node" type and return the intent and slot TP, FP, FN counts.
    Optionally collect the per label TP, FP, FN counts.
    """

    predicted_intents_and_slots = _get_intents_and_slots(
        predicted_frame, tree_based=tree_based
    )
    expected_intents_and_slots = _get_intents_and_slots(
        expected_frame, tree_based=tree_based
    )
    return IntentSlotConfusions(
        intent_confusions=_compare_nodes(
            predicted_intents_and_slots.intents,
            expected_intents_and_slots.intents,
            intent_per_label_confusions,
        ),
        slot_confusions=_compare_nodes(
            predicted_intents_and_slots.slots,
            expected_intents_and_slots.slots,
            slot_per_label_confusions,
        ),
    )


# TODO: split this file, move compute_classification_metrics() (bottom) to a separate
#   file, and rename this function to compute_classification_metrics().
def compute_classification_metrics_from_nodes_pairs(
    nodes_pairs: List[NodesPredictionPair]
) -> Tuple[AllConfusions, AllClassificationMetrics]:
    """
    Compute classification metrics given a list of pairs of predicted and expected
    node sets.
    """
    all_confusions = AllConfusions()
    for (predicted_nodes, expected_nodes) in nodes_pairs:
        all_confusions.confusions += _compare_nodes(
            predicted_nodes, expected_nodes, all_confusions.per_label_confusions
        )
    return all_confusions, all_confusions.compute_metrics()


def compute_intent_slot_metrics(
    frame_pairs: List[FramePredictionPair],
    tree_based: bool,
    overall_metrics: bool = True,
) -> IntentSlotMetrics:
    """
    Given a list of (predicted_frame, expected_frame) tuples in Node type,
    return IntentSlotMetrics. We take the following assumptions:
    1. The root Node is intent.
    2. Children of intents are always slots, and children of slots are always intents.
    The input parameter tree_based determines whether bracket (tree_based=False) or
    tree (tree_based=True) scores are computed.
    """
    intents_pairs: List[NodesPredictionPair] = []
    slots_pairs: List[NodesPredictionPair] = []
    for (predicted_frame, expected_frame) in frame_pairs:
        predicted = _get_intents_and_slots(predicted_frame, tree_based=tree_based)
        expected = _get_intents_and_slots(expected_frame, tree_based=tree_based)
        intents_pairs.append(NodesPredictionPair(predicted.intents, expected.intents))
        slots_pairs.append(NodesPredictionPair(predicted.slots, expected.slots))
    intent_confusions, intent_metrics = compute_classification_metrics_from_nodes_pairs(
        intents_pairs
    )
    slot_confusions, slot_metrics = compute_classification_metrics_from_nodes_pairs(
        slots_pairs
    )

    return IntentSlotMetrics(
        intent_metrics=intent_metrics,
        slot_metrics=slot_metrics,
        overall_metrics=(
            intent_confusions.confusions + slot_confusions.confusions
        ).compute_metrics()
        if overall_metrics
        else None,
    )


def compute_top_intent_accuracy(frame_pairs: List[FramePredictionPair]) -> float:
    num_correct = 0
    num_samples = len(frame_pairs)
    for (predicted_frame, expected_frame) in frame_pairs:
        num_correct += int(predicted_frame.label == expected_frame.label)
    return _safe_division(num_correct, num_samples)


def compute_frame_accuracy(frame_pairs: List[FramePredictionPair]) -> float:
    num_correct = 0
    num_samples = len(frame_pairs)
    for (predicted_frame, expected_frame) in frame_pairs:
        num_correct += int(predicted_frame == expected_frame)
    return _safe_division(num_correct, num_samples)


def compute_frame_accuracies_by_depth(
    frame_pairs: List[FramePredictionPair]
) -> FrameAccuraciesByDepth:
    """
    Given a list of (predicted_frame, expected_frame) pairs of Node type, split the
    prediction pairs into buckets according to the tree depth of the expected frame, and
    compute frame accuracies for each bucket.
    """
    frame_pairs_by_depth: Dict[int, List[FramePredictionPair]] = defaultdict(list)
    for frame_pair in frame_pairs:
        depth = frame_pair.expected_frame.get_depth()
        frame_pairs_by_depth[depth].append(frame_pair)
    frame_accuracies_by_depth: FrameAccuraciesByDepth = {}
    for depth, pairs in frame_pairs_by_depth.items():
        frame_accuracies_by_depth[depth] = FrameAccuracy(
            len(pairs), compute_frame_accuracy(pairs)
        )
    return frame_accuracies_by_depth


def compute_all_metrics(
    frame_pairs: List[FramePredictionPair],
    top_intent_accuracy: bool = True,
    frame_accuracy: bool = True,
    frame_accuracies_by_depth: bool = True,
    bracket_metrics: bool = True,
    tree_metrics: bool = True,
    overall_metrics: bool = False,
) -> AllMetrics:
    """
    Given a list of (predicted_frame, expected_frame) pairs of Node type, return both
    bracket and tree metrics.
    """
    top_intent = (
        compute_top_intent_accuracy(frame_pairs) if top_intent_accuracy else None
    )
    accuracy = compute_frame_accuracy(frame_pairs) if frame_accuracy else None
    accuracies = (
        compute_frame_accuracies_by_depth(frame_pairs)
        if frame_accuracies_by_depth
        else None
    )
    bracket = (
        compute_intent_slot_metrics(
            frame_pairs, tree_based=False, overall_metrics=overall_metrics
        )
        if bracket_metrics
        else None
    )
    tree = (
        compute_intent_slot_metrics(
            frame_pairs, tree_based=True, overall_metrics=overall_metrics
        )
        if tree_metrics
        else None
    )

    return AllMetrics(top_intent, accuracy, accuracies, bracket, tree)


def compute_classification_metrics(
    label_pairs: List[LabelPredictionPair],
    label_names: List[str],
    label_scores: Optional[List[LabelScoresPair]] = None,
) -> Union[AllClassificationMetrics, BinaryClassificationMetrics]:
    """
    A general function that computes classification metrics given a list of pairs of
    predicted and expected labels.
    """
    all_confusions = AllConfusions()
    for (predicted_label, expected_label) in label_pairs:
        if predicted_label == expected_label:
            all_confusions.confusions.TP += 1
            all_confusions.per_label_confusions.update(expected_label, "TP", 1)
        else:
            all_confusions.confusions.FP += 1
            all_confusions.confusions.FN += 1
            all_confusions.per_label_confusions.update(expected_label, "FN", 1)
            all_confusions.per_label_confusions.update(predicted_label, "FP", 1)
    metrics = all_confusions.compute_metrics()
    if label_scores is not None and len(label_names) == 2:
        soft_metrics = compute_soft_metrics(label_scores, label_names)
        confusion_dict = all_confusions.per_label_confusions.dictionary
        # since MCC is symmetric, it doesn't matter which label is 0 and which is 1
        TP = confusion_dict[label_names[0]].TP
        FP = confusion_dict[label_names[0]].FP
        FN = confusion_dict[label_names[0]].FN
        TN = confusion_dict[label_names[1]].TP
        mcc = compute_matthews_correlation_coefficients(TP=TP, FP=FP, FN=FN, TN=TN)
        binary_metrics = BinaryClassificationMetrics(
            all_classification_metrics=metrics,
            per_label_soft_scores=soft_metrics,
            mcc=mcc,
        )
        return binary_metrics
    return metrics


def compute_soft_metrics(
    label_scores: List[LabelScoresPair], label_names: List[str]
) -> Dict[str, SoftClassificationMetrics]:
    """
    Computes classification metrics given a list of pairs of
    scores and expected labels.
    """

    soft_metrics = {}
    for i, label_name in enumerate(label_names):
        y_true = []
        y_score = []
        for (predicted_scores, expected_label) in label_scores:
            y_true.append(expected_label == label_name)
            y_score.append(predicted_scores[i])
        ap = average_precision_score(y_true, y_score)
        soft_metrics[label_name] = SoftClassificationMetrics(average_precision=ap)
    return soft_metrics


def compute_matthews_correlation_coefficients(
    TP: int, FP: int, TN: int, FN: int
) -> float:
    """
    Matthews correlation coefficient is a way to summarize all four values of
    the confusin matrix for binary classification.
    """
    mcc = _safe_division(
        (TP * TN) - (FP * FN), np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
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

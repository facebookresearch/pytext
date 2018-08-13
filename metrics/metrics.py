#!/usr/bin/env python3

import collections
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


class IntentsAndSlots(NamedTuple):
    intents: Counter[Node]
    slots: Counter[Node]


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
    per_label_scores and macro_scores because they are both computed on a per label
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

    def print_metrics(self):
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


class IntentSlotMetrics(NamedTuple):
    """
    Intent and slot metrics.
    """

    intent_metrics: AllClassificationMetrics
    slot_metrics: AllClassificationMetrics
    overall_metrics: ClassificationMetrics

    def print_metrics(self):
        print("\nIntent Metrics")
        self.intent_metrics.print_metrics()
        print("\nSlot Metrics")
        self.slot_metrics.print_metrics()
        print("\nMerging Intent and Slot Metrics")
        print(
            f"  P = {self.overall_metrics.precision * 100:.2f} "
            f"R = {self.overall_metrics.recall * 100:.2f}, "
            f"F1 = {self.overall_metrics.f1 * 100:.2f}.\n"
        )


# TODO: add frame accuracy to the metrics
class AllMetrics:
    __slots__ = (
        "intent_bracket_scores",
        "intent_tree_scores",
        "slot_bracket_scores",
        "slot_tree_scores",
    )

    def __init__(
        self,
        intent_bracket_scores: Optional[AllClassificationMetrics] = None,
        intent_tree_scores: Optional[AllClassificationMetrics] = None,
        slot_bracket_scores: Optional[AllClassificationMetrics] = None,
        slot_tree_scores: Optional[AllClassificationMetrics] = None,
    ) -> None:
        self.intent_bracket_scores = intent_bracket_scores
        self.intent_tree_scores = intent_tree_scores
        self.slot_bracket_scores = slot_bracket_scores
        self.slot_tree_scores = slot_tree_scores


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
        self.dictionary: DefaultDict[str, Confusions] = collections.defaultdict(
            Confusions
        )

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
    intents: Counter[Node] = collections.Counter()
    slots: Counter[Node] = collections.Counter()

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


def compute_intent_slot_metrics(
    frame_pairs: List[Tuple[Node, Node]], tree_based: bool
) -> IntentSlotMetrics:
    """
    Given a list of (predicted_frame, expected_frame) tuples in Node type,
    return IntentSlotMetrics. We take the following assumptions:
    1. The root Node is intent.
    2. Children of intents are always slots, and children of slots are always intents.
    The input parameter tree_based determines whether bracket (tree_based=False) or
    tree (tree_based=True) scores are computed.
    """

    intent_confusions, slot_confusions = Confusions(), Confusions()
    intent_per_label_confusions, slot_per_label_confusions = (
        PerLabelConfusions(),
        PerLabelConfusions(),
    )
    for (predicted_frame, expected_frame) in frame_pairs:
        intent_slot_confusions = compare_frames(
            predicted_frame,
            expected_frame,
            tree_based=tree_based,
            intent_per_label_confusions=intent_per_label_confusions,
            slot_per_label_confusions=slot_per_label_confusions,
        )
        intent_confusions += intent_slot_confusions.intent_confusions
        slot_confusions += intent_slot_confusions.slot_confusions

    intent_metrics = intent_per_label_confusions.compute_metrics()
    slot_metrics = slot_per_label_confusions.compute_metrics()
    return IntentSlotMetrics(
        intent_metrics=AllClassificationMetrics(
            per_label_scores=intent_metrics.per_label_scores,
            macro_scores=intent_metrics.macro_scores,
            micro_scores=intent_confusions.compute_metrics(),
        ),
        slot_metrics=AllClassificationMetrics(
            per_label_scores=slot_metrics.per_label_scores,
            macro_scores=slot_metrics.macro_scores,
            micro_scores=slot_confusions.compute_metrics(),
        ),
        overall_metrics=(intent_confusions + slot_confusions).compute_metrics(),
    )


def compute_all_metrics(
    frame_pairs: List[Tuple[Node, Node]], bracket: bool = True, tree: bool = True
) -> AllMetrics:
    """
    Given a list of (predicted_frame, expected_frame) tuples in Node type,
    return both bracket and tree metrics.
    """

    if bracket:
        bracket_metrics = compute_intent_slot_metrics(frame_pairs, tree_based=False)
    if tree:
        tree_metrics = compute_intent_slot_metrics(frame_pairs, tree_based=True)

    return AllMetrics(
        bracket_metrics.intent_metrics if bracket else None,
        tree_metrics.intent_metrics if tree else None,
        bracket_metrics.slot_metrics if bracket else None,
        tree_metrics.slot_metrics if tree else None,
    )

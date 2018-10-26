#!/usr/bin/env python3

from collections import Counter as counter, defaultdict
from copy import deepcopy
from typing import Any, Counter, Dict, List, NamedTuple, Optional, Set, Tuple

from .basic_metrics import (
    AllConfusions,
    Confusions,
    PerLabelConfusions,
    PRF1Metrics,
    PRF1Scores,
    safe_division,
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

    def get_depth(self) -> int:
        return 1 + max((child.get_depth() for child in self.children), default=0)


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


class IntentSlotMetrics(NamedTuple):
    """
    Intent and slot metrics.
    """

    intent_metrics: Optional[PRF1Metrics]
    slot_metrics: Optional[PRF1Metrics]
    overall_metrics: Optional[PRF1Scores]

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


class AllMetrics(NamedTuple):
    top_intent_accuracy: Optional[float]
    frame_accuracy: Optional[float]
    frame_accuracies_by_depth: Optional[FrameAccuraciesByDepth]
    bracket_metrics: Optional[IntentSlotMetrics]
    tree_metrics: Optional[IntentSlotMetrics]

    def print_metrics(self) -> None:
        if self.frame_accuracy:
            print(f"\n\nFrame accuracy = {self.frame_accuracy * 100:.2f}")
        if self.bracket_metrics:
            print("\n\nBracket Metrics")
            self.bracket_metrics.print_metrics()
        if self.tree_metrics:
            print("\n\nTree Metrics")
            self.tree_metrics.print_metrics()


class IntentSlotConfusions(NamedTuple):
    intent_confusions: Confusions
    slot_confusions: Confusions


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


def compute_prf1_metrics(
    nodes_pairs: List[NodesPredictionPair]
) -> Tuple[AllConfusions, PRF1Metrics]:
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
    intent_confusions, intent_metrics = compute_prf1_metrics(intents_pairs)
    slot_confusions, slot_metrics = compute_prf1_metrics(slots_pairs)

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
    return safe_division(num_correct, num_samples)


def compute_frame_accuracy(frame_pairs: List[FramePredictionPair]) -> float:
    num_correct = 0
    num_samples = len(frame_pairs)
    for (predicted_frame, expected_frame) in frame_pairs:
        num_correct += int(predicted_frame == expected_frame)
    return safe_division(num_correct, num_samples)


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

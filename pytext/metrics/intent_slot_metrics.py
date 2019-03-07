#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from collections import Counter as counter, defaultdict
from copy import deepcopy
from typing import (
    AbstractSet,
    Any,
    Counter,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

from pytext.data.data_structures.node import Node as NodeBase, Span

from . import (
    AllConfusions,
    Confusions,
    PerLabelConfusions,
    PRF1Metrics,
    PRF1Scores,
    safe_division,
)


"""
Metric classes and functions for intent-slot prediction problems.
"""


class Node(NodeBase):
    """
    Subclass of the base Node class, used for metric purposes. It is immutable so that
    hashing can be done on the class.

    Attributes:
        label (str): Label of the node.
        span (Span): Span of the node.
        children (:obj:`frozenset` of :obj:`Node`): frozenset of the node's children,
            left empty when computing bracketing metrics.
    """

    def __init__(
        self, label: str, span: Span, children: Optional[AbstractSet["Node"]] = None
    ) -> None:
        super().__init__(label, span, frozenset(children) if children else frozenset())

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Node class is immutable.")

    def __hash__(self):
        return hash((self.label, self.span))


class FramePredictionPair(NamedTuple):
    """
    Pair of predicted and gold intent frames.
    """

    predicted_frame: Node
    expected_frame: Node


class NodesPredictionPair(NamedTuple):
    """
    Pair of predicted and expected sets of nodes.
    """

    predicted_nodes: Counter[Node]
    expected_nodes: Counter[Node]


class IntentsAndSlots(NamedTuple):
    """
    Collection of intents and slots in an intent frame.
    """

    intents: Counter[Node]
    slots: Counter[Node]


class FrameAccuracy(NamedTuple):
    """
    Frame accuracy for a collection of intent frame predictions.

    Frame accuracy means the entire tree structure of the predicted frame matches that
    of the gold frame.
    """

    num_samples: int
    frame_accuracy: float


FrameAccuraciesByDepth = Dict[int, FrameAccuracy]
"""
Frame accuracies bucketized by depth of the gold tree.
"""


class IntentSlotMetrics(NamedTuple):
    """
    Precision/recall/F1 metrics for intents and slots.

    Attributes:
        intent_metrics: Precision/recall/F1 metrics for intents.
        slot_metrics: Precision/recall/F1 metrics for slots.
        overall_metrics: Combined precision/recall/F1 metrics for all nodes (merging
            intents and slots).
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
    """
    Aggregated class for intent-slot related metrics.

    Attributes:
        top_intent_accuracy: Accuracy of the top-level intent.
        frame_accuracy: Frame accuracy.
        frame_accuracies_by_depth: Frame accuracies bucketized by depth of the gold
            tree.
        bracket_metrics: Bracket metrics for intents and slots. For details, see the
            function `compute_intent_slot_metrics()`.
        tree_metrics: Tree metrics for intents and slots. For details, see the function
            `compute_intent_slot_metrics()`.
        loss: Cross entropy loss.
    """

    top_intent_accuracy: Optional[float]
    frame_accuracy: Optional[float]
    frame_accuracy_top_k: Optional[float]
    frame_accuracies_by_depth: Optional[FrameAccuraciesByDepth]
    bracket_metrics: Optional[IntentSlotMetrics]
    tree_metrics: Optional[IntentSlotMetrics]
    loss: Optional[float] = None

    def print_metrics(self) -> None:
        if self.frame_accuracy:
            print(f"\n\nFrame accuracy = {self.frame_accuracy * 100:.2f}")
        if self.frame_accuracy_top_k:
            print(f"\n\nTop k frame accuracy = {self.frame_accuracy_top_k * 100:.2f}")
        if self.bracket_metrics:
            print("\n\nBracket Metrics")
            self.bracket_metrics.print_metrics()
        if self.tree_metrics:
            print("\n\nTree Metrics")
            self.tree_metrics.print_metrics()


class IntentSlotConfusions(NamedTuple):
    """
    Aggregated class for intent and slot confusions.

    Attributes:
        intent_confusions: Confusion counts for intents.
        slot_confusions: Confusion counts for slots.
    """

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
    Compares two intent frames and returns TP, FP, FN counts for intents and slots.
    Optionally collects the per label TP, FP, FN counts.

    Args:
        predicted_frame: Predicted intent frame.
        expected_frame: Gold intent frame.
        tree_based: Whether to get the tree-based confusions (if True) or bracket-based
            confusions (if False). For details, see the function
            `compute_intent_slot_metrics()`.
        intent_per_label_confusions: If provided, update the per label confusions for
            intents as well. Defaults to None.
        slot_per_label_confusions: If provided, update the per label confusions for
            slots as well. Defaults to None.

    Returns:
        IntentSlotConfusions, containing confusion counts for intents and slots.
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
    nodes_pairs: Sequence[NodesPredictionPair]
) -> Tuple[AllConfusions, PRF1Metrics]:
    """
    Computes precision/recall/F1 metrics given a list of predicted and expected sets of
    nodes.

    Args:
        nodes_pairs: List of predicted and expected node sets.

    Returns:
        A tuple, of which the first member contains the confusion information, and the
        second member contains the computed precision/recall/F1 metrics.
    """
    all_confusions = AllConfusions()
    for (predicted_nodes, expected_nodes) in nodes_pairs:
        all_confusions.confusions += _compare_nodes(
            predicted_nodes, expected_nodes, all_confusions.per_label_confusions
        )
    return all_confusions, all_confusions.compute_metrics()


def compute_intent_slot_metrics(
    frame_pairs: Sequence[FramePredictionPair],
    tree_based: bool,
    overall_metrics: bool = True,
) -> IntentSlotMetrics:
    """
    Given a list of predicted and gold intent frames, computes precision, recall and F1
    metrics for intents and slots, either in tree-based or bracket-based manner.

    The following assumptions are taken on intent frames:
    1. The root node is an intent,
    2. Children of intents are always slots, and children of slots are always intents.

    For tree-based metrics, a node (an intent or slot) in the predicted frame is
    considered a true positive only if the subtree rooted at this node has an exact copy
    in the gold frame, otherwise it is considered a false positive. A false negative is
    a node in the gold frame that does not have an exact subtree match in the predicted
    frame.

    For bracket-based metrics, a node in the predicted frame is considered a true
    positive if there is a node in the gold frame having the same label and span (but
    not necessarily the same children). The definitions of false positives and false
    negatives are similar to the above.

    Args:
        frame_pairs: List of predicted and gold intent frames.
        tree_based: Whether to compute tree-based metrics (if True) or bracket-based
            metrics (if False).
        overall_metrics: Whether to compute overall (merging intents and slots) metrics
            or not. Defaults to True.

    Returns:
        IntentSlotMetrics, containing precision/recall/F1 metrics for intents and slots.
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


def compute_top_intent_accuracy(frame_pairs: Sequence[FramePredictionPair]) -> float:
    """
    Computes accuracy of the top-level intent.

    Args:
        frame_pairs: List of predicted and gold intent frames.

    Returns:
        Prediction accuracy of the top-level intent.
    """
    num_correct = 0
    num_samples = len(frame_pairs)
    for (predicted_frame, expected_frame) in frame_pairs:
        num_correct += int(predicted_frame.label == expected_frame.label)
    return safe_division(num_correct, num_samples)


def compute_frame_accuracy(frame_pairs: Sequence[FramePredictionPair]) -> float:
    """
    Computes frame accuracy given a list of predicted and gold intent frames.

    Args:
        frame_pairs: List of predicted and gold intent frames.

    Returns:
        Frame accuracy. For a prediction, frame accuracy is achieved if the entire tree
        structure of the predicted frame matches that of the gold frame.
    """
    num_correct = 0
    num_samples = len(frame_pairs)
    for (predicted_frame, expected_frame) in frame_pairs:
        num_correct += int(predicted_frame == expected_frame)
    return safe_division(num_correct, num_samples)


def compute_frame_accuracy_top_k(
    frame_pairs: List[FramePredictionPair], all_frames: List[List[Node]]
) -> Tuple[float, int]:
    num_samples = len(frame_pairs)
    num_correct = 0
    for i, top_k_predicted_frames in enumerate(all_frames):
        _, expected_frame = frame_pairs[i]
        for predicted_frame in top_k_predicted_frames:
            if predicted_frame == expected_frame:
                num_correct += 1
                break
    return safe_division(num_correct, num_samples)


def compute_frame_accuracies_by_depth(
    frame_pairs: Sequence[FramePredictionPair]
) -> FrameAccuraciesByDepth:
    """
    Given a list of predicted and gold intent frames, splits the predictions into
    buckets according to the depth of the gold trees, and computes frame accuracy for
    each bucket.

    Args:
        frame_pairs: List of predicted and gold intent frames.

    Returns:
        FrameAccuraciesByDepth, a map from depths to their corresponding frame
        accuracies.
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
    frame_pairs: Sequence[FramePredictionPair],
    top_intent_accuracy: bool = True,
    frame_accuracy: bool = True,
    frame_accuracies_by_depth: bool = True,
    bracket_metrics: bool = True,
    tree_metrics: bool = True,
    overall_metrics: bool = False,
    all_predicted_frames: List[List[Node]] = None,
    calculated_loss: float = None,
) -> AllMetrics:
    """
    Given a list of predicted and gold intent frames, computes intent-slot related
    metrics.

    Args:
        frame_pairs: List of predicted and gold intent frames.
        top_intent_accuracy: Whether to compute top intent accuracy or not. Defaults to
            True.
        frame_accuracy: Whether to compute frame accuracy or not. Defaults to True.
        frame_accuracies_by_depth: Whether to compute frame accuracies by depth or not.
            Defaults to True.
        bracket_metrics: Whether to compute bracket metrics or not. Defaults to True.
        tree_metrics: Whether to compute tree metrics or not. Defaults to True.
        overall_metrics: If `bracket_metrics` or `tree_metrics` is true, decides whether
            to compute overall (merging intents and slots) metrics for them. Defaults to
            False.

    Returns:
        AllMetrics which contains intent-slot related metrics.
    """
    frame_accuracy_top_k = 0
    if all_predicted_frames:
        frame_accuracy_top_k = compute_frame_accuracy_top_k(
            frame_pairs, all_predicted_frames
        )

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

    return AllMetrics(
        top_intent,
        accuracy,
        frame_accuracy_top_k,
        accuracies,
        bracket,
        tree,
        calculated_loss,
    )

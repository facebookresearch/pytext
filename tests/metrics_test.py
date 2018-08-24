#!/usr/bin/env python3

from typing import Any, Dict, List

from pytext.metrics import (
    AllClassificationMetrics,
    ClassificationMetrics,
    Confusions,
    IntentSlotConfusions,
    IntentSlotMetrics,
    MacroClassificationMetrics,
    Node,
    Span,
    compare_frames,
    compute_all_metrics,
    compute_intent_slot_metrics,
)

from .metrics_test_base import MetricsTestBase


TEST_EXAMPLES: List[Dict[str, Any]] = [
    # Non-nested examples
    {  # Two identical frames
        "predicted": Node(
            label="intent1",
            span=Span(start=0, end=20),
            children={Node(label="slot1", span=Span(start=1, end=2))},
        ),
        "expected": Node(
            label="intent1",
            span=Span(start=0, end=20),
            children={Node(label="slot1", span=Span(start=1, end=2))},
        ),
        "frames_match": True,
        "bracket_confusions": {
            "intent_confusion": {"TP": 1, "FP": 0, "FN": 0},
            "slot_confusion": {"TP": 1, "FP": 0, "FN": 0},
        },
        "tree_confusions": {
            "intent_confusion": {"TP": 1, "FP": 0, "FN": 0},
            "slot_confusion": {"TP": 1, "FP": 0, "FN": 0},
        },
    },
    {  # One frame is missing slots
        "predicted": Node(label="intent1", span=Span(start=0, end=20)),
        "expected": Node(
            label="intent1",
            span=Span(start=0, end=20),
            children={Node(label="slot1", span=Span(start=1, end=2))},
        ),
        "frames_match": False,
        "bracket_confusions": {
            "intent_confusion": {"TP": 1, "FP": 0, "FN": 0},
            "slot_confusion": {"TP": 0, "FP": 0, "FN": 1},
        },
        "tree_confusions": {
            "intent_confusion": {"TP": 0, "FP": 1, "FN": 1},
            "slot_confusion": {"TP": 0, "FP": 0, "FN": 1},
        },
    },
    {  # A wrong intent
        "predicted": Node(
            label="intent1",
            span=Span(start=0, end=20),
            children={Node(label="slot1", span=Span(start=1, end=2))},
        ),
        "expected": Node(
            label="intent2",
            span=Span(start=0, end=20),
            children={Node(label="slot2", span=Span(start=1, end=2))},
        ),
        "frames_match": False,
        "bracket_confusions": {
            "intent_confusion": {"TP": 0, "FP": 1, "FN": 1},
            "slot_confusion": {"TP": 0, "FP": 1, "FN": 1},
        },
        "tree_confusions": {
            "intent_confusion": {"TP": 0, "FP": 1, "FN": 1},
            "slot_confusion": {"TP": 0, "FP": 1, "FN": 1},
        },
    },
    {  # Overlapping slots
        "predicted": Node(
            label="intent1",
            span=Span(start=0, end=20),
            children={Node(label="slot1", span=Span(start=1, end=3))},
        ),
        "expected": Node(
            label="intent1",
            span=Span(start=0, end=20),
            children={Node(label="slot1", span=Span(start=2, end=4))},
        ),
        "frames_match": False,
        "bracket_confusions": {
            "intent_confusion": {"TP": 1, "FP": 0, "FN": 0},
            "slot_confusion": {"TP": 0, "FP": 1, "FN": 1},
        },
        "tree_confusions": {
            "intent_confusion": {"TP": 0, "FP": 1, "FN": 1},
            "slot_confusion": {"TP": 0, "FP": 1, "FN": 1},
        },
    },
    {  # Non-overlapping slots
        "predicted": Node(
            label="intent1",
            span=Span(start=0, end=20),
            children={Node(label="slot1", span=Span(start=1, end=3))},
        ),
        "expected": Node(
            label="intent1",
            span=Span(start=0, end=20),
            children={Node(label="slot1", span=Span(start=6, end=7))},
        ),
        "frames_match": False,
        "bracket_confusions": {
            "intent_confusion": {"TP": 1, "FP": 0, "FN": 0},
            "slot_confusion": {"TP": 0, "FP": 1, "FN": 1},
        },
        "tree_confusions": {
            "intent_confusion": {"TP": 0, "FP": 1, "FN": 1},
            "slot_confusion": {"TP": 0, "FP": 1, "FN": 1},
        },
    },
    {  # One matched and one missed slot
        "predicted": Node(
            label="intent1",
            span=Span(start=0, end=20),
            children={Node(label="slot2", span=Span(start=10, end=12))},
        ),
        "expected": Node(
            label="intent1",
            span=Span(start=0, end=20),
            children={
                Node(label="slot1", span=Span(start=6, end=7)),
                Node(label="slot2", span=Span(start=10, end=12)),
            },
        ),
        "frames_match": False,
        "bracket_confusions": {
            "intent_confusion": {"TP": 1, "FP": 0, "FN": 0},
            "slot_confusion": {"TP": 1, "FP": 0, "FN": 1},
        },
        "tree_confusions": {
            "intent_confusion": {"TP": 0, "FP": 1, "FN": 1},
            "slot_confusion": {"TP": 1, "FP": 0, "FN": 1},
        },
    },
    {  # True slot covers two predicted slots
        "predicted": Node(
            label="intent1",
            span=Span(start=0, end=20),
            children={
                Node(label="slot1", span=Span(start=2, end=5)),
                Node(label="slot2", span=Span(start=6, end=10)),
                Node(label="slot3", span=Span(start=11, end=20)),
            },
        ),
        "expected": Node(
            label="intent1",
            span=Span(start=0, end=20),
            children={
                Node(label="slot1", span=Span(start=1, end=11)),
                Node(label="slot3", span=Span(start=11, end=20)),
            },
        ),
        "frames_match": False,
        "bracket_confusions": {
            "intent_confusion": {"TP": 1, "FP": 0, "FN": 0},
            "slot_confusion": {"TP": 1, "FP": 2, "FN": 1},
        },
        "tree_confusions": {
            "intent_confusion": {"TP": 0, "FP": 1, "FN": 1},
            "slot_confusion": {"TP": 1, "FP": 2, "FN": 1},
        },
    },
    {  # One matched and one wrong slot
        "predicted": Node(
            label="intent1",
            span=Span(start=0, end=20),
            children={
                Node(label="slot1", span=Span(start=1, end=10)),
                Node(label="slot2", span=Span(start=11, end=20)),
            },
        ),
        "expected": Node(
            label="intent1",
            span=Span(start=0, end=20),
            children={Node(label="slot1", span=Span(start=1, end=10))},
        ),
        "frames_match": False,
        "bracket_confusions": {
            "intent_confusion": {"TP": 1, "FP": 0, "FN": 0},
            "slot_confusion": {"TP": 1, "FP": 1, "FN": 0},
        },
        "tree_confusions": {
            "intent_confusion": {"TP": 0, "FP": 1, "FN": 1},
            "slot_confusion": {"TP": 1, "FP": 1, "FN": 0},
        },
    },
    # Nested examples
    {  # Two identical frames
        "predicted": Node(
            label="intent1",
            span=Span(start=0, end=20),
            children={
                Node(
                    label="slot1",
                    span=Span(start=1, end=5),
                    children={
                        Node(
                            label="intent2",
                            span=Span(start=1, end=5),
                            children={Node(label="slot2", span=Span(start=1, end=2))},
                        )
                    },
                )
            },
        ),
        "expected": Node(
            label="intent1",
            span=Span(start=0, end=20),
            children={
                Node(
                    label="slot1",
                    span=Span(start=1, end=5),
                    children={
                        Node(
                            label="intent2",
                            span=Span(start=1, end=5),
                            children={Node(label="slot2", span=Span(start=1, end=2))},
                        )
                    },
                )
            },
        ),
        "frames_match": True,
        "bracket_confusions": {
            "intent_confusion": {"TP": 2, "FP": 0, "FN": 0},
            "slot_confusion": {"TP": 2, "FP": 0, "FN": 0},
        },
        "tree_confusions": {
            "intent_confusion": {"TP": 2, "FP": 0, "FN": 0},
            "slot_confusion": {"TP": 2, "FP": 0, "FN": 0},
        },
    },
    {  # One nested and one not
        "predicted": Node(
            label="intent1",
            span=Span(start=0, end=20),
            children={
                Node(
                    label="slot1",
                    span=Span(start=1, end=5),
                    children={
                        Node(
                            label="intent2",
                            span=Span(start=1, end=5),
                            children={
                                Node(label="slot2", span=Span(start=1, end=2)),
                                Node(label="slot3", span=Span(start=4, end=5)),
                            },
                        )
                    },
                )
            },
        ),
        "expected": Node(
            label="intent1",
            span=Span(start=0, end=20),
            children={Node(label="slot1", span=Span(start=1, end=5))},
        ),
        "frames_match": False,
        "bracket_confusions": {
            "intent_confusion": {"TP": 1, "FP": 1, "FN": 0},
            "slot_confusion": {"TP": 1, "FP": 2, "FN": 0},
        },
        "tree_confusions": {
            "intent_confusion": {"TP": 0, "FP": 2, "FN": 1},
            "slot_confusion": {"TP": 0, "FP": 3, "FN": 1},
        },
    },
    {  # More complex example
        "predicted": Node(
            label="intent1",
            span=Span(start=0, end=20),
            children={
                Node(
                    label="slot1",
                    span=Span(start=1, end=6),
                    children={
                        Node(
                            label="intent2",
                            span=Span(start=1, end=6),
                            children={
                                Node(label="slot4", span=Span(start=4, end=6)),
                                Node(label="slot3", span=Span(start=1, end=2)),
                            },
                        )
                    },
                ),
                Node(label="slot2", span=Span(start=7, end=10)),
            },
        ),
        "expected": Node(
            label="intent1",
            span=Span(start=0, end=20),
            children={
                Node(
                    label="slot1",
                    span=Span(start=1, end=5),
                    children={
                        Node(
                            label="intent2",
                            span=Span(start=1, end=5),
                            children={
                                Node(label="slot3", span=Span(start=1, end=2)),
                                Node(label="slot4", span=Span(start=4, end=5)),
                            },
                        )
                    },
                ),
                Node(label="slot2", span=Span(start=7, end=10)),
            },
        ),
        "frames_match": False,
        "bracket_confusions": {
            "intent_confusion": {"TP": 1, "FP": 1, "FN": 1},
            "slot_confusion": {"TP": 2, "FP": 2, "FN": 2},
        },
        "tree_confusions": {
            "intent_confusion": {"TP": 0, "FP": 2, "FN": 2},
            "slot_confusion": {"TP": 2, "FP": 2, "FN": 2},
        },
    },
]


class MetricsTest(MetricsTestBase):
    def test_compare_frames(self) -> None:
        i = 0
        for example in TEST_EXAMPLES:
            self.assertEqual(
                compare_frames(
                    example["predicted"], example["expected"], tree_based=False
                ),
                IntentSlotConfusions(
                    intent_confusions=Confusions(
                        **example["bracket_confusions"]["intent_confusion"]
                    ),
                    slot_confusions=Confusions(
                        **example["bracket_confusions"]["slot_confusion"]
                    ),
                ),
                i,
            )
            self.assertEqual(
                compare_frames(
                    example["predicted"], example["expected"], tree_based=True
                ),
                IntentSlotConfusions(
                    intent_confusions=Confusions(
                        **example["tree_confusions"]["intent_confusion"]
                    ),
                    slot_confusions=Confusions(
                        **example["tree_confusions"]["slot_confusion"]
                    ),
                ),
            )
            i += 1

    def test_compute_intent_slot_metrics(self) -> None:
        # Test single pair
        self.assertMetricsAlmostEqual(
            compute_intent_slot_metrics(
                [(TEST_EXAMPLES[1]["predicted"], TEST_EXAMPLES[1]["expected"])],
                tree_based=False,
            ),
            IntentSlotMetrics(
                intent_metrics=AllClassificationMetrics(
                    per_label_scores={
                        "intent1": ClassificationMetrics(1, 0, 0, 1.0, 1.0, 1.0)
                    },
                    macro_scores=MacroClassificationMetrics(1, 1.0, 1.0, 1.0),
                    micro_scores=ClassificationMetrics(1, 0, 0, 1.0, 1.0, 1.0),
                ),
                slot_metrics=AllClassificationMetrics(
                    per_label_scores={
                        "slot1": ClassificationMetrics(0, 0, 1, 0.0, 0.0, 0.0)
                    },
                    macro_scores=MacroClassificationMetrics(1, 0.0, 0.0, 0.0),
                    micro_scores=ClassificationMetrics(0, 0, 1, 0.0, 0.0, 0.0),
                ),
                overall_metrics=ClassificationMetrics(1, 0, 1, 1.0, 0.5, 2.0 / 3),
            ),
        )
        self.assertMetricsAlmostEqual(
            compute_intent_slot_metrics(
                [(TEST_EXAMPLES[1]["predicted"], TEST_EXAMPLES[1]["expected"])],
                tree_based=True,
            ),
            IntentSlotMetrics(
                intent_metrics=AllClassificationMetrics(
                    per_label_scores={
                        "intent1": ClassificationMetrics(0, 1, 1, 0.0, 0.0, 0.0)
                    },
                    macro_scores=MacroClassificationMetrics(1, 0.0, 0.0, 0.0),
                    micro_scores=ClassificationMetrics(0, 1, 1, 0.0, 0.0, 0.0),
                ),
                slot_metrics=AllClassificationMetrics(
                    per_label_scores={
                        "slot1": ClassificationMetrics(0, 0, 1, 0.0, 0.0, 0.0)
                    },
                    macro_scores=MacroClassificationMetrics(1, 0.0, 0.0, 0.0),
                    micro_scores=ClassificationMetrics(0, 0, 1, 0.0, 0.0, 0.0),
                ),
                overall_metrics=ClassificationMetrics(0, 1, 2, 0.0, 0.0, 0.0),
            ),
        )

        # Test multiple pairs consisting of the 8th through the 11th examples
        pairs = []
        for i in range(7, 11):
            pairs.append((TEST_EXAMPLES[i]["predicted"], TEST_EXAMPLES[i]["expected"]))

        self.assertMetricsAlmostEqual(
            compute_intent_slot_metrics(pairs, tree_based=False),
            IntentSlotMetrics(
                intent_metrics=AllClassificationMetrics(
                    per_label_scores={
                        # intent1: TP = 4, FP = 0, FN = 0
                        "intent1": ClassificationMetrics(4, 0, 0, 1.0, 1.0, 1.0),
                        # intent2: TP = 1, FP = 2, FN = 1
                        "intent2": ClassificationMetrics(1, 2, 1, 1.0 / 3, 0.5, 0.4),
                    },
                    macro_scores=MacroClassificationMetrics(2, 2.0 / 3, 0.75, 0.7),
                    # all intents: TP = 5, FP = 2, FN = 1
                    micro_scores=ClassificationMetrics(
                        5, 2, 1, 5.0 / 7, 5.0 / 6, 10.0 / 13
                    ),
                ),
                slot_metrics=AllClassificationMetrics(
                    per_label_scores={
                        # slot1: TP = 3, FP = 1, FN = 1
                        "slot1": ClassificationMetrics(3, 1, 1, 0.75, 0.75, 0.75),
                        # slot2: TP = 2, FP = 2, FN = 0
                        "slot2": ClassificationMetrics(2, 2, 0, 0.5, 1.0, 4.0 / 6),
                        # slot3: TP = 1, FP = 1, FN = 0
                        "slot3": ClassificationMetrics(1, 1, 0, 0.5, 1.0, 2.0 / 3),
                        # slot4: TP = 0, FP = 1, FN = 1
                        "slot4": ClassificationMetrics(0, 1, 1, 0.0, 0.0, 0.0),
                    },
                    macro_scores=MacroClassificationMetrics(
                        4, 0.4375, 0.6875, 25.0 / 48
                    ),
                    # all slots: TP = 6, FP = 5, FN = 2
                    micro_scores=ClassificationMetrics(
                        6, 5, 2, 6.0 / 11, 6.0 / 8, 12.0 / 19
                    ),
                ),
                # overall: TP = 11, FP = 7, FN = 3
                overall_metrics=ClassificationMetrics(
                    11, 7, 3, 11.0 / 18, 11.0 / 14, 22.0 / 32
                ),
            ),
        )

        self.assertMetricsAlmostEqual(
            compute_intent_slot_metrics(pairs, tree_based=True),
            IntentSlotMetrics(
                intent_metrics=AllClassificationMetrics(
                    per_label_scores={
                        # intent1: TP = 1, FP = 3, FN = 3
                        "intent1": ClassificationMetrics(1, 3, 3, 0.25, 0.25, 0.25),
                        # intent2: TP = 1, FP = 2, FN = 1
                        "intent2": ClassificationMetrics(1, 2, 1, 1.0 / 3, 0.5, 0.4),
                    },
                    macro_scores=MacroClassificationMetrics(2, 7.0 / 24, 0.375, 0.325),
                    # all intents: TP = 2, FP = 5, FN = 4
                    micro_scores=ClassificationMetrics(
                        2, 5, 4, 2.0 / 7, 2.0 / 6, 4.0 / 13
                    ),
                ),
                slot_metrics=AllClassificationMetrics(
                    per_label_scores={
                        # slot1: TP = 2, FP = 2, FN = 2
                        "slot1": ClassificationMetrics(2, 2, 2, 0.5, 0.5, 0.5),
                        # slot2: TP = 2, FP = 2, FN = 0
                        "slot2": ClassificationMetrics(2, 2, 0, 0.5, 1.0, 4.0 / 6),
                        # slot3: TP = 1, FP = 1, FN = 0
                        "slot3": ClassificationMetrics(1, 1, 0, 0.5, 1.0, 2.0 / 3),
                        # slot4: TP = 0, FP = 1, FN = 1
                        "slot4": ClassificationMetrics(0, 1, 1, 0.0, 0.0, 0.0),
                    },
                    macro_scores=MacroClassificationMetrics(4, 0.375, 0.625, 11.0 / 24),
                    # all slots: TP = 5, FP = 6, FN = 3
                    micro_scores=ClassificationMetrics(
                        5, 6, 3, 5.0 / 11, 5.0 / 8, 10.0 / 19
                    ),
                ),
                # overall: TP = 7, FP = 11, FN = 7
                overall_metrics=ClassificationMetrics(
                    7, 11, 7, 7.0 / 18, 7.0 / 14, 14.0 / 32
                ),
            ),
        )

    # Just to test the metrics print without errors
    def test_print_compute_all_metrics(self) -> None:
        frame_pairs = [
            (example["predicted"], example["expected"]) for example in TEST_EXAMPLES
        ]
        compute_all_metrics(frame_pairs, overall_metrics=True).print_metrics()

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from unittest import TestCase

from pytext.data.data_structures.annotation import Annotation
from pytext.metric_reporters.compositional_metric_reporter import (
    CompositionalMetricReporter,
)
from pytext.metrics.intent_slot_metrics import Node, Span


def get_frame(parse: str) -> Node:
    annotation = Annotation(parse)
    frame = CompositionalMetricReporter.tree_to_metric_node(annotation.tree)
    return frame


class TestCompositionalMetricReporter(TestCase):
    def test_tree_to_metric_node(self):
        TEXT_EXAMPLES = [
            (
                "[IN:alarm/set_alarm  repeat the [SL:datetime 3 : 00 pm ] "
                + "[SL:alarm/name alarm ]  [SL:datetime for Sunday august 12th ]  ] ",
                Node(
                    label="IN:alarm/set_alarm",
                    span=Span(start=0, end=49),
                    children={
                        Node(
                            label="SL:datetime",
                            span=Span(
                                start=11,
                                end=20,
                            ),
                            text="3 : 00 pm",
                        ),
                        Node(
                            label="SL:alarm/name",
                            span=Span(
                                start=21,
                                end=26,
                            ),
                            text="alarm",
                        ),
                        Node(
                            label="SL:datetime",
                            span=Span(
                                start=27,
                                end=49,
                            ),
                            text="for Sunday august 12th",
                        ),
                    },
                    text="repeat the",
                ),
            ),
            (
                "[IN:calling/call_friend call [SL:person moms ] cellphone ]",
                Node(
                    label="IN:calling/call_friend",
                    span=Span(start=0, end=19),
                    children={
                        Node(label="SL:person", span=Span(start=5, end=9), text="moms")
                    },
                    text="call cellphone",
                ),
            ),
            (
                "[IN:GET_DIRECTIONS I need [SL:ANCHOR directions] to [SL:DESTINATION "
                + "[IN:GET_EVENT the jazz festival]]]",
                Node(
                    label="IN:GET_DIRECTIONS",
                    span=Span(start=0, end=38),
                    text="I need to",
                    children={
                        Node(
                            label="SL:ANCHOR",
                            span=Span(start=7, end=17),
                            text="directions",
                        ),
                        Node(
                            label="SL:DESTINATION",
                            span=Span(start=21, end=38),
                            text="",
                            children={
                                Node(
                                    label="IN:GET_EVENT",
                                    span=Span(start=21, end=38),
                                    text="the jazz festival",
                                )
                            },
                        ),
                    },
                ),
            ),
        ]
        for annotation_string, expected_frame in TEXT_EXAMPLES:
            frame = get_frame(annotation_string)
            self.assertEqual(frame, expected_frame)

    def test_no_match(self):
        FAILURE_CASES = [
            (
                "[IN:CREATE_CALL [SL:CONTACT john ] ]",
                "[IN:CREATE_CALL [SL:CONTACT jane ] ]",
            )
        ]

        for annotation_string_1, annotation_string_2 in FAILURE_CASES:
            frame1 = get_frame(annotation_string_1)
            frame2 = get_frame(annotation_string_2)
            self.assertNotEqual(frame1, frame2)

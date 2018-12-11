#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from unittest import TestCase

from pytext.data.data_structures.annotation import Annotation
from pytext.metric_reporters.compositional_metric_reporter import (
    CompositionalMetricReporter,
)
from pytext.metrics.intent_slot_metrics import Node, Span


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
                        Node(label="SL:datetime", span=Span(start=11, end=20)),
                        Node(label="SL:alarm/name", span=Span(start=21, end=26)),
                        Node(label="SL:datetime", span=Span(start=27, end=49)),
                    },
                ),
            ),
            (
                "[IN:calling/call_friend call [SL:person moms ] cellphone ]",
                Node(
                    label="IN:calling/call_friend",
                    span=Span(start=0, end=19),
                    children={Node(label="SL:person", span=Span(start=5, end=9))},
                ),
            ),
            (
                "[IN:GET_DIRECTIONS I need [SL:ANCHOR directions] to [SL:DESTINATION "
                + "[IN:GET_EVENT the jazz festival]]]",
                Node(
                    label="IN:GET_DIRECTIONS",
                    span=Span(start=0, end=38),
                    children={
                        Node(label="SL:ANCHOR", span=Span(start=7, end=17)),
                        Node(
                            label="SL:DESTINATION",
                            span=Span(start=21, end=38),
                            children={
                                Node(label="IN:GET_EVENT", span=Span(start=21, end=38))
                            },
                        ),
                    },
                ),
            ),
        ]
        for annotation_string, expected_frame in TEXT_EXAMPLES:
            annotation = Annotation(annotation_string)
            frame = CompositionalMetricReporter.tree_to_metric_node(annotation.tree)
            self.assertEqual(frame, expected_frame)

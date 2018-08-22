#!/usr/bin/env python3
from libfb.py import testutil

from messenger.assistant.cu.python.utils.metrics import frames_are_equal
from pytext import metrics
from pytext.rnng.annotation import Annotation
from pytext.shared_tokenizer import SharedTokenizer
from pytext.rnng.tools.annotation_to_intent_frame import (
    annotation_to_intent_frame,
    intent_frame_to_tree,
    tree_to_intent_frame,
    tree_to_metric_node,
)
from messenger.assistant.cu.core.ttypes import IntentFrame, FilledSlot, Span


class TestAnnotationToIntentFrame(testutil.BaseFacebookTestCase):
    def get_flat_data(self, case=0):
        if case == 0:
            utterance = "repeat the 3:00pm alarm for Sunday august 12th"
            annotation_string = (
                "[IN:alarm/set_alarm    "
                + "repeat the [SL:datetime 3 : 00 pm ]"
                + "  [SL:alarm/name alarm ]  [SL:datetime for Sunday august 12th ]  ] "
            )
            annotation_token_spans = SharedTokenizer().tokenize_with_ranges(utterance)
            annotation = Annotation(annotation_string)

            intent_frame = IntentFrame(
                utterance=utterance,
                domain="alarm",
                intent="IN:alarm/set_alarm",
                slots=[
                    FilledSlot(
                        id="SL:datetime", span=Span(start=11, end=17), text="3:00pm"
                    ),
                    FilledSlot(
                        id="SL:alarm/name", span=Span(start=18, end=23), text="alarm"
                    ),
                    FilledSlot(
                        id="SL:datetime",
                        span=Span(start=24, end=46),
                        text="for Sunday august 12th",
                    ),
                ],
                span=Span(start=0, end=46),
            )
        elif case == 1:
            utterance = "call moms cellphone"
            annotation_string = (
                "[IN:calling/call_friend call [SL:person moms ] cellphone ]"
            )
            annotation_token_spans = SharedTokenizer().tokenize_with_ranges(utterance)
            annotation = Annotation(annotation_string)

            intent_frame = IntentFrame(
                utterance=utterance,
                domain="calling",
                intent="IN:calling/call_friend",
                slots=[
                    FilledSlot(id="SL:person", span=Span(start=5, end=8), text="mom")
                ],
                span=Span(start=0, end=19),
            )
        elif case == 2:
            utterance = "repeat the 3:00pm alarm for Sunday august 12th"
            annotation_string = (
                "[alarm/set_alarm    "
                + "repeat the [datetime 3 : 00 pm ]"
                + "  [alarm/name alarm ]  [datetime for Sunday august 12th ]  ] "
            )
            annotation_token_spans = SharedTokenizer().tokenize_with_ranges(
                utterance
            )
            annotation = Annotation(
                annotation_string,
                accept_flat_intents_slots=True
            )

            intent_frame = IntentFrame(
                utterance=utterance,
                domain="alarm",
                intent="alarm/set_alarm",
                slots=[
                    FilledSlot(
                        id="datetime",
                        span=Span(start=11, end=17),
                        text="3:00pm"
                    ),
                    FilledSlot(
                        id="alarm/name",
                        span=Span(start=18, end=23),
                        text="alarm"
                    ),
                    FilledSlot(
                        id="datetime",
                        span=Span(start=24, end=46),
                        text="for Sunday august 12th",
                    ),
                ],
                span=Span(start=0, end=46),
            )

        return annotation, utterance, annotation_token_spans, intent_frame

    def test_flat_annotation_to_intent_frame(self):
        test_annotation, test_utterance, test_annotation_token_spans, test_intent_frame = self.get_flat_data()

        frame = annotation_to_intent_frame(
            test_annotation, test_utterance, test_annotation_token_spans, domain="alarm"
        )

        self.assertEqual(frame, test_intent_frame)

    def test_flat_annotation_to_intent_frame_accepting_flat_labels(self):
        test_annotation, test_utterance, \
            test_annotation_token_spans, \
            test_intent_frame = self.get_flat_data(2)

        frame = annotation_to_intent_frame(
            test_annotation, test_utterance, test_annotation_token_spans, domain="alarm"
        )

        self.assertEqual(frame, test_intent_frame)

    def test_flat_intent_frame_to_tree(self):
        for case in range(2):
            test_annotation, test_utterance, test_annotation_token_spans, test_intent_frame = self.get_flat_data(
                case
            )

            tree = intent_frame_to_tree(test_intent_frame)

            self.assertEqual(str(tree), str(test_annotation))

    def test_flat_tree_to_metric_node(self):
        for case in range(2):
            _, _, _, test_intent_frame = self.get_flat_data(case)

            if case == 0:
                test_frame = metrics.Node(
                    label="IN:alarm/set_alarm",
                    span=metrics.Span(start=0, end=49),
                    children={
                        metrics.Node(
                            label="SL:datetime", span=metrics.Span(start=11, end=20)
                        ),
                        metrics.Node(
                            label="SL:alarm/name", span=metrics.Span(start=21, end=26)
                        ),
                        metrics.Node(
                            label="SL:datetime",
                            span=metrics.Span(start=27, end=49),
                        ),
                    },
                )
            elif case == 1:
                test_frame = metrics.Node(
                    label="IN:calling/call_friend",
                    span=metrics.Span(start=0, end=19),
                    children={
                        metrics.Node(
                            label="SL:person", span=metrics.Span(start=5, end=9)
                        )
                    },
                )

            tree = intent_frame_to_tree(test_intent_frame)
            frame = tree_to_metric_node(tree)

            self.assertEqual(frame, test_frame)

    def get_compositional_data(self):
        utterance = "I need directions to the jazz festival"
        annotation_string = (
            "[IN:GET_DIRECTIONS I need [SL:ANCHOR directions] to [SL:DESTINATION"
            + " [IN:GET_EVENT the jazz festival]]]"
        )
        annotation_token_spans = SharedTokenizer().tokenize_with_ranges(utterance)
        annotation = Annotation(annotation_string)

        intent_frame = IntentFrame(
            utterance=utterance,
            domain="navigation",
            intent="IN:GET_DIRECTIONS",
            slots=[
                FilledSlot(
                    id="SL:ANCHOR", span=Span(start=7, end=17), text="directions"
                ),
                FilledSlot(
                    id="SL:DESTINATION",
                    span=Span(start=21, end=38),
                    text="the jazz festival",
                    subframe=IntentFrame(
                        domain="",
                        utterance="the jazz festival",
                        intent="IN:GET_EVENT",
                        slots=[],
                        span=Span(start=0, end=17),
                    ),
                ),
            ],
            span=Span(start=0, end=38),
        )

        return annotation, utterance, annotation_token_spans, intent_frame

    def test_compositional_annotation_to_intent_frame(self):
        test_annotation, test_utterance, test_annotation_token_spans, test_intent_frame = (
            self.get_compositional_data()
        )

        frame = annotation_to_intent_frame(
            test_annotation, test_utterance, test_annotation_token_spans, domain="navigation"
        )

        self.assertEqual(frame, test_intent_frame)

    def test_compositional_intent_frame_to_tree(self):
        test_annotation, test_utterance, test_annotation_token_spans, test_intent_frame = (
            self.get_compositional_data()
        )

        tree = intent_frame_to_tree(test_intent_frame)

        self.assertEqual(str(tree), str(test_annotation))

    def test_compositional_tree_to_metric_node(self):
        _, _, _, test_intent_frame = self.get_compositional_data()
        test_frame = metrics.Node(
            label="IN:GET_DIRECTIONS",
            span=metrics.Span(start=0, end=38),
            children={
                metrics.Node(label="SL:ANCHOR", span=metrics.Span(start=7, end=17)),
                metrics.Node(
                    label="SL:DESTINATION",
                    span=metrics.Span(start=21, end=38),
                    children={
                        metrics.Node(
                            label="IN:GET_EVENT", span=metrics.Span(start=21, end=38)
                        )
                    },
                ),
            },
        )

        tree = intent_frame_to_tree(test_intent_frame)
        frame = tree_to_metric_node(tree)

        self.assertEqual(frame, test_frame)

    # TODO: (wenfangxu) T32687283 remove this unit test when we do not depend on
    #       tree_to_intent_frame()
    def test_compositional_tree_to_intent_frame(self):
        _, _, _, test_intent_frame = self.get_compositional_data()

        tree = intent_frame_to_tree(test_intent_frame)
        intent_frame = tree_to_intent_frame(tree)

        self.assertEqual(frames_are_equal(intent_frame, test_intent_frame), True)

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from unittest import TestCase

from pytext.metric_reporters.intent_slot_detection_metric_reporter import (
    create_frame,
    frame_to_str,
)
from pytext.utils.data import byte_length


class TestIntentSlotMetricReporter(TestCase):
    def test_create_node(self):
        TEXT_EXAMPLES = [
            ("exit", "device/close_app", "", "[device/close_app exit ]"),
            (
                "call mom",
                "IN:CREATE_CALL",
                "5:8:SL:CONTACT",
                "[IN:CREATE_CALL call [SL:CONTACT mom ] ]",
            ),
            (
                "An Yu",
                "meta/provideSlotValue",
                "0:5:SL:CONTACT",
                "[meta/provideSlotValue [SL:CONTACT An Yu ] ]",
            ),
            (
                "Set a reminder to pick up Sean at 3:15 pm today.",
                "IN:CREATE_REMINDER",
                "18:30:SL:TODO,34:47:SL:DATE_TIME",
                "[IN:CREATE_REMINDER Set a reminder to [SL:TODO pick up Sean ] "
                "at [SL:DATE_TIME 3:15 pm today ]. ]",
            ),
            (
                "Set a reminder to pick up Sean at 3:15 pm today.",
                "IN:CREATE_REMINDER",
                "34:47:SL:DATE_TIME,18:30:SL:TODO",
                "[IN:CREATE_REMINDER Set a reminder to [SL:TODO pick up Sean ] "
                "at [SL:DATE_TIME 3:15 pm today ]. ]",
            ),
            ('["Fine"]', "cu:other", "", r'[cu:other \["Fine"\] ]'),
            (  # Example in byte offset.
                "establece el escándalo a 7",
                "IN:SET_VOLUME",
                "26:27:SL:PRECISE_AMOUNT",
                "[IN:SET_VOLUME establece el escándalo a [SL:PRECISE_AMOUNT 7 ] ]",
            ),
        ]
        for (
            utterance,
            intent_label,
            slot_names_str,
            expected_annotation_str,
        ) in TEXT_EXAMPLES:
            frame = create_frame(
                utterance, intent_label, slot_names_str, byte_length(utterance)
            )
            self.assertEqual(frame_to_str(frame), expected_annotation_str)

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from unittest import TestCase

from pytext.metric_reporters.intent_slot_detection_metric_reporter import (
    IntentSlotChannel,
)


class TestIntentSlotMetricReporter(TestCase):
    def test_create_annotation(self):
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
            ('["Fine"]', "cu:other", "", r'[cu:other \["Fine"\] ]'),
        ]
        for (
            utterance,
            intent_label,
            slot_names_str,
            expected_annotation_str,
        ) in TEXT_EXAMPLES:
            annotation_str = IntentSlotChannel.create_annotation(
                utterance, intent_label, slot_names_str
            )
            self.assertEqual(annotation_str, expected_annotation_str)

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from unittest import TestCase

from pytext.data.data_structures.annotation import Annotation


class TestAnnotation(TestCase):
    def test_annotation(self):
        TEST_EXAMPLES = [
            ("[device/close_app exit ]", None),
            ("[IN:CREATE_CALL call [SL:CONTACT mom ] ]", None),
            ("[meta/provideSlotValue [SL:CONTACT An Yu ] ]", None),
            (
                "[IN:CREATE_REMINDER Set a reminder to [SL:TODO pick up Sean ] "
                "at [SL:DATE_TIME 3:15 pm today ]. ]",
                "[IN:CREATE_REMINDER Set a reminder to [SL:TODO pick up Sean ] "
                "at [SL:DATE_TIME 3:15 pm today ] . ]",
            ),
            (
                "[IN:CREATE_REMINDER Remind me to [SL:TODO [IN:CREATE_CALL "
                "[SL:METHOD_CALL call ] [SL:CONTACT John ] ] ] [SL:DATE_TIME at 6 pm "
                "tonight ] ]",
                None,
            ),
            (  # The same example above with some whitespaces removed
                "[IN:CREATE_REMINDER Remind me to[SL:TODO[IN:CREATE_CALL"
                "[SL:METHOD_CALL call][SL:CONTACT John]]][SL:DATE_TIME at 6 pm "
                "tonight]]",
                "[IN:CREATE_REMINDER Remind me to [SL:TODO [IN:CREATE_CALL "
                "[SL:METHOD_CALL call ] [SL:CONTACT John ] ] ] [SL:DATE_TIME at 6 pm "
                "tonight ] ]",
            ),
            (  # Combination labels
                "[IN:GET_INFO_TRAFFIC [SL:OBSTRUCTION Traffic ] please? ] "
                "[IN:GET_ESTIMATED_DURATION How long is my [SL:METHOD_TRAVEL drive ] "
                "[SL:DESTINATION [IN:GET_LOCATION_HOME home ] ] ]",
                "[IN:COMBINE [SL:COMBINE [IN:GET_INFO_TRAFFIC [SL:OBSTRUCTION Traffic ]"
                " please? ] ] [SL:COMBINE [IN:GET_ESTIMATED_DURATION How long is my "
                "[SL:METHOD_TRAVEL drive ] [SL:DESTINATION [IN:GET_LOCATION_HOME home ]"
                " ] ] ] ]",
            ),
            (r'[cu:other \["Fine"\] ]', None),  # Annotation uses escape
        ]
        for annotation_str, expected_annotation_str in TEST_EXAMPLES:
            expected_annotation_str = expected_annotation_str or annotation_str
            annotation = Annotation(annotation_str, accept_flat_intents_slots=True)
            self.assertEqual(
                annotation.tree.flat_str().strip(), expected_annotation_str
            )

    def test_annotation_errors(self):
        """
        Test invalid annotation strings for which the Annotation class should raise
        ValueError.
        """

        TEST_EXAMPLES = (
            # Extra brackets
            "[device/close_app please [] exit ]",
            # Missing closing bracket
            "[IN:CREATE_CALL call [SL:CONTACT mom ]",
            # Missing intent label
            "[IN:CREATE_REMINDER Remind me to [ [IN:CREATE_CALL [SL:METHOD_CALL call ] "
            "[SL:CONTACT John ] ] ] [SL:DATE_TIME at 6 pm tonight ] ]",
            # No brackets
            "hang on, it's marty's party, not mary's party",
        )
        for annotation_str in TEST_EXAMPLES:
            try:
                Annotation(annotation_str, accept_flat_intents_slots=True)
            except ValueError as e:
                print(e)
                pass
            else:
                raise Exception("Annotation error not catched.")

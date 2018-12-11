#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

from pytext.fields.field import FloatVectorField


class FieldTest(unittest.TestCase):
    def setUp(self):
        self.float_vector = FloatVectorField(10)

    def test_float_vector_array_style(self):
        tensor = self.float_vector.process(
            [
                self.float_vector.preprocess(
                    "[0.64840776,0.7575,0.5531,0.2403,0,0.9481,0,0.1538,0.2403]"
                )
            ]
        )
        self.assertEqual(tensor[0][0], 0.64840776)
        self.assertEqual(tensor[0][9], 0.0)  # Test padding

    def test_float_vector_array_plain(self):
        tensor = self.float_vector.process(
            [
                self.float_vector.preprocess(
                    "0.64840776 0.7575,0.5531,    0.2403,0,0.9481,0,0.1538,0.2403"
                )
            ]
        )
        self.assertEqual(tensor[0][0], 0.64840776)
        self.assertEqual(tensor[0][1], 0.7575)  # Test space separation
        self.assertEqual(tensor[0][2], 0.5531)  # Test comma separation
        self.assertEqual(tensor[0][9], 0.0)  # Test padding

    def test_float_vector_array_not_closed(self):
        tensor = self.float_vector.process(
            [
                self.float_vector.preprocess(
                    "[, 0.64840776 0.7575,0.5531,    0.2403,0,0.9481,0,0.1538,0.2403"
                )
            ]
        )
        self.assertEqual(tensor[0][0], 0.64840776)
        self.assertEqual(tensor[0][1], 0.7575)  # Test space separation
        self.assertEqual(tensor[0][2], 0.5531)  # Test comma separation
        self.assertEqual(tensor[0][9], 0.0)  # Test padding

    def test_float_vector_array_not_opened(self):
        tensor = self.float_vector.process(
            [
                self.float_vector.preprocess(
                    "[, 0.64840776 0.7575,0.5531,    0.2403,0,0.9481,0,0.1538,0.2403]"
                )
            ]
        )
        self.assertEqual(tensor[0][0], 0.64840776)
        self.assertEqual(tensor[0][1], 0.7575)  # Test space separation
        self.assertEqual(tensor[0][2], 0.5531)  # Test comma separation
        self.assertEqual(tensor[0][9], 0.0)  # Test padding

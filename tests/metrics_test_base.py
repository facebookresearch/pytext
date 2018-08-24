#!/usr/bin/env python3

from typing import Any
from unittest import TestCase


class MetricsTestBase(TestCase):
    def assertMetricsAlmostEqual(self, first: Any, second: Any) -> None:
        self.assertEqual(type(first), type(second))
        if isinstance(first, int):
            self.assertEqual(first, second)
        elif isinstance(first, float):
            self.assertAlmostEqual(first, second)
        elif isinstance(first, dict):
            self.assertEqual(first.keys(), second.keys())
            for key in first.keys():
                self.assertMetricsAlmostEqual(first[key], second[key])
        # Then "first" and "second" should either be of type NamedTuple which has
        # a _fields field or they should have __slots__ field.
        else:
            for attr in getattr(first, "_fields", None) or first.__slots__:
                self.assertMetricsAlmostEqual(
                    getattr(first, attr), getattr(second, attr)
                )

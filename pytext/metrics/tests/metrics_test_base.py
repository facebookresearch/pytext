#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Any
from unittest import TestCase


class MetricsTestBase(TestCase):
    def assertMetricsAlmostEqual(self, first: Any, second: Any) -> None:
        self.assertEqual(type(first), type(second))
        if first is None:
            return
        elif isinstance(first, int):
            self.assertEqual(first, second)
        elif isinstance(first, float):
            self.assertAlmostEqual(first, second)
        elif isinstance(first, dict):
            self.assertEqual(first.keys(), second.keys())
            for key in first.keys():
                self.assertMetricsAlmostEqual(first[key], second[key])
        # Then "first" and "second" should be of type NamedTuple.
        else:
            self.assertEqual(first._fields, second._fields)
            for attr in first._fields:
                self.assertMetricsAlmostEqual(
                    getattr(first, attr), getattr(second, attr)
                )

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

from pytext.utils import timing


class TimingTest(unittest.TestCase):
    def test_format_time(self):
        tests = (
            (1.2473e-8, "0.0ns"),
            (1.2473e-7, "0.1ns"),
            (1.2473e-6, "1.2ns"),
            (0.000_012_473, "12.5ns"),
            (0.000_124_73, "124.7ns"),
            (0.001_247_3, "1.2ms"),
            (0.012_473, "12.5ms"),
            (0.12473, "124.7ms"),
            (1.2473, "1.2s"),
            (12.473, "12.5s"),
            (124.73, "2m5s"),
            (1247.3, "20m47s"),
            (12473.0, "3h28m"),
            (124_730.0, "1d11h"),
            (1_247_300.0, "14d10h"),
        )
        for seconds, expected in tests:
            self.assertEqual(
                expected, timing.format_time(seconds), f"Failed to format {seconds}"
            )

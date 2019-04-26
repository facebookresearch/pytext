#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

from pytext.utils import meters


class MetersTest(unittest.TestCase):
    def test_time_meter(self):
        tps = meters.TimeMeter()
        for i in range(10):
            tps.update(i)
        self.assertEqual(tps.n, 55)
        self.assertTrue(tps.avg > 1)

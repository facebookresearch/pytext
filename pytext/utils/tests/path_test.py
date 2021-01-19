#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

from pytext.utils.path import is_absolute_path


class PathTest(unittest.TestCase):
    def test_is_absolute_path(self):
        self.assertEqual(is_absolute_path("/mnt/vol/pytext/encoder.pt"), True)
        self.assertEqual(is_absolute_path("manifold://pytext/tree/encoder.pt"), True)
        self.assertEqual(is_absolute_path("encoder.pt"), False)

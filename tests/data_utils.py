#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os


TEST_BASE_DIR = os.environ.get(
    "PYTEXT_TEST_DATA", os.path.join(os.path.dirname(__file__), "data")
)


def test_file(filename):
    return os.path.join(TEST_BASE_DIR, filename)

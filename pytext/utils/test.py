#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from importlib import import_module


def import_tests_module(packages_to_scan=None):
    if not packages_to_scan:
        packages_to_scan = ["pytext.tests", "tests"]

    for package in packages_to_scan:
        try:
            return import_module(".data_utils", package=package)
        except (ModuleNotFoundError, SystemError):
            pass
    else:
        raise ModuleNotFoundError(f"Scanned packages: {packages_to_scan}")

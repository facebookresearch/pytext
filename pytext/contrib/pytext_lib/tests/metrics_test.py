#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

import hydra
from pytext.contrib.pytext_lib.conf import ClassificationMetricReporterConf
from pytext.metric_reporters import ClassificationMetricReporter


class TestMetrics(unittest.TestCase):
    def test_classification_metric_reporter_hydra_init_default(self):
        default_mr = ClassificationMetricReporterConf()
        classification_mr = hydra.utils.call(default_mr, label_names=["0", "1"])
        self.assertIsInstance(classification_mr, ClassificationMetricReporter)

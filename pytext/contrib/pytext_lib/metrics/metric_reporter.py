#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List

from pytext.metric_reporters import ClassificationMetricReporter


def classification_metric_reporter_config_expand(label_names: List[str], **kwargs):
    classification_metric_reporter_config = ClassificationMetricReporter.Config(
        **kwargs
    )
    return ClassificationMetricReporter.from_config_and_label_names(
        classification_metric_reporter_config, label_names
    )

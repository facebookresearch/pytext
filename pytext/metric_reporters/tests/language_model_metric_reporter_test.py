#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from unittest import TestCase
from unittest.mock import patch

import torch
from pytext.common.constants import Stage
from pytext.config.module_config import PerplexityType
from pytext.metric_reporters.language_model_metric_reporter import (
    MaskedLMMetricReporter,
)


class TestMaskedLMMetricReporter(TestCase):
    def test_report_realtime_metric(self):
        class MockMetadata:
            class MockTarget:
                pad_token_idx = 0

            def __init__(self):
                self.target = self.MockTarget()

        class MockTrainState:
            def __init__(self):
                self.batch_counter = 512
                self.step_counter = 128

        reporter = MaskedLMMetricReporter(
            channels=[],
            metadata=MockMetadata(),
            tensorizers=None,
            aggregate_metrics=True,
            perplexity_type=PerplexityType.MEDIAN,
            pep_format=False,
        )
        reporter.add_batch_stats(
            n_batches=128,
            preds=None,
            targets=(None, torch.tensor([1, 1]), torch.tensor([1, 1, 1, 1])),
            scores=None,
            loss=torch.tensor([2.4]),
            m_input=None,
        )

        def mock_print(*args, **kwargs):
            metrics = {}
            for metric in args[0].split(","):
                key, val = metric.split(":")
                metrics[key.strip()] = float(val.strip())

            # math.exp(loss) ==> e^2.4 = 11.02
            self.assertEqual(metrics["batch ppl"], 11.02)
            self.assertEqual(metrics["agg ppl"], 11.02)
            self.assertEqual(metrics["number of batches"], 128)

        with patch("builtins.print", side_effect=mock_print):
            reporter.report_realtime_metric(Stage.TRAIN)

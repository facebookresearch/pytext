#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from unittest import TestCase
from unittest.mock import patch

import torch
from pytext.common.constants import Stage
from pytext.config.module_config import PerplexityType
from pytext.metric_reporters.language_model_metric_reporter import (
    LanguageModelMetricReporter,
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


class TestLMMetricsReporter(TestCase):
    def test_report_lm_all_target_equals_padding(self):
        class MockMetadata:
            class MockTarget:
                pad_token_idx = 1

            def __init__(self):
                self.target = self.MockTarget()

        metrics_reporter = LanguageModelMetricReporter(
            channels=[],
            metadata=MockMetadata(),
            tensorizers=None,
            aggregate_metrics=True,
            perplexity_type=PerplexityType.MEDIAN,
            pep_format=False,
        )
        # this is a corner case that both the actual target and
        # the padding are all 1, creating a target that will
        # be dropped from score calculation entirely
        target_words = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
        target_lens = torch.tensor([3, 2, 5])
        targets = (target_words, target_lens)
        emb_dim = 128
        preds = torch.rand((target_words.shape[0], target_words.shape[1], emb_dim))
        metrics_reporter.add_batch_stats(
            n_batches=128,
            preds=preds,
            targets=targets,
            scores=None,
            loss=torch.tensor([1]),
            m_input=None,
        )
        self.assertEqual(len(metrics_reporter.all_scores), 0)

    def test_report_empty_batch(self):
        class MockMetadata:
            class MockTarget:
                pad_token_idx = 1

            def __init__(self):
                self.target = self.MockTarget()

        metrics_reporter = LanguageModelMetricReporter(
            channels=[],
            metadata=MockMetadata(),
            tensorizers=None,
            aggregate_metrics=True,
            perplexity_type=PerplexityType.MEDIAN,
            pep_format=False,
        )
        target_words = torch.tensor([])
        target_lens = torch.tensor([0])
        targets = (target_words, target_lens)
        preds = torch.tensor([])
        metrics_reporter.add_batch_stats(
            n_batches=128,
            preds=preds,
            targets=targets,
            scores=None,
            loss=torch.tensor([1]),
            m_input=None,
        )
        self.assertEqual(len(metrics_reporter.all_scores), 0)

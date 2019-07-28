#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from unittest import TestCase

import torch
from pytext.config.module_config import PerplexityType
from pytext.metric_reporters.language_model_metric_reporter import (
    MaskedLMMetricReporter,
)


class TestMaskedLMMetricReporter(TestCase):
    def test_report_logging(self):
        class MockMetadata:
            class MockTarget:
                pad_token_idx = 0

            def __init__(self):
                self.target = self.MockTarget()

        reporter = MaskedLMMetricReporter(
            channels=[],
            metadata=MockMetadata(),
            tensorizers=None,
            aggregate_metrics=True,
            perplexity_type=PerplexityType.MEDIAN,
        )
        reporter.add_batch_stats(
            n_batches=128,
            preds=None,
            targets=(None, torch.tensor([1, 1]), torch.tensor([1, 1, 1, 1])),
            scores=None,
            loss=torch.tensor([2.4]),
            m_input=None,
        )
        trainer_logging = {"n_updates": 128, "n_batches": 512}
        metric_logging = reporter.get_metric_logging()

        metric_keys = ("tps", "ppl", "batch_ppl", "batch_tps")
        for key in metric_keys:
            self.assertTrue(key in metric_logging)

        # loss x targets[1].sum() ==> 2.4 x 2
        self.assertEqual(round(reporter.aggregate_loss, 1), 4.8)
        self.assertEqual(reporter.total_num_tokens, 2)
        # math.exp(ppl) ==> math.exp(2.4) ~ 11.02
        self.assertEqual(round(metric_logging["ppl"], 2), 11.02)

        logging = reporter.aggregate_logging([metric_logging], [trainer_logging])
        for key in metric_keys:
            self.assertTrue(key in logging)

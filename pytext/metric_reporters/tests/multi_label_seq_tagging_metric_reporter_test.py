#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from unittest import TestCase

import torch
from pytext.metric_reporters.word_tagging_metric_reporter import (
    MultiLabelSequenceTaggingMetricReporter,
)
from pytext.metrics import MultiLabelSoftClassificationMetrics


class TestMultiLabelSequenceTaggingMetricReporter(TestCase):
    def test_calculate_metric(self):

        metrics_reporter = MultiLabelSequenceTaggingMetricReporter(
            label_names=["set1", "set2"],
            pad_idx=[-1, -1],
            channels=[],
            label_vocabs=[["l1", "l2"], ["s1", "s2", "s3"]],
        )
        # shape of the tensors: bsz * turn
        targets = (
            torch.tensor([[0, 1, 0], [0, 1, 0], [1, -1, -1]]),
            torch.tensor([[0, 2, 1], [0, 1, 2], [0, 1, -1]]),
        )
        preds = (
            torch.tensor([[0, 0, 0], [0, 1, 0], [1, 1, 1]]),
            torch.tensor([[0, 1, 1], [0, 1, 2], [1, 1, 2]]),
        )
        scores = (torch.rand(3, 3, 2), torch.rand(3, 3, 3))

        metrics_reporter.add_batch_stats(
            n_batches=1,
            preds=preds,
            targets=targets,
            scores=scores,
            loss=None,
            # just to mock the batch size
            m_input=([0, 0, 0],),
        )
        metrics_reporter.add_batch_stats(
            n_batches=1,
            preds=preds,
            targets=targets,
            scores=scores,
            loss=None,
            # just to mock the batch size
            m_input=([0, 0, 0],),
        )
        self.assertEqual(len(metrics_reporter.all_preds), 6)
        self.assertEqual(len(metrics_reporter.all_preds[0]), 2)
        self.assertEqual(len(metrics_reporter.all_preds[0][1]), 3)
        self.assertEqual(len(metrics_reporter.all_scores), 6, "should have 6 examples")
        self.assertEqual(
            len(metrics_reporter.all_scores[0]), 2, "example1 should have 2 sets"
        )
        self.assertEqual(
            len(metrics_reporter.all_scores[0][0]),
            3,
            "example1 set1, should have 3 turns",
        )
        self.assertEqual(
            len(metrics_reporter.all_scores[0][0][0]),
            2,
            "example1 set1, should have 2 scores (2 labels)",
        )

        self.assertEqual(
            len(metrics_reporter.all_scores[0][1][0]),
            3,
            "example1 set2, should have 3 scores (3 labels)",
        )
        metrics = metrics_reporter.calculate_metric()
        self.assertTrue(isinstance(metrics, MultiLabelSoftClassificationMetrics))

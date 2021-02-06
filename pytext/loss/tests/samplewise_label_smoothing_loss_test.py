#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

import torch
import torch.nn.functional as F
from pytext.loss import LabelSmoothingLoss, SamplewiseLabelSmoothingLoss


class SamplewiseLabelSmoothingLossTest(unittest.TestCase):
    def test_samplewise_label_smoothing_loss(self):
        batch_size = 5
        num_labels = 5

        label_smoothing_loss = LabelSmoothingLoss(
            LabelSmoothingLoss.Config(), ignore_index=-1
        )
        samplewise_label_smoothing_loss = SamplewiseLabelSmoothingLoss(
            SamplewiseLabelSmoothingLoss.Config(), ignore_index=-1
        )

        logits = F.log_softmax(torch.rand(batch_size, num_labels), 1)
        targets = torch.randint(batch_size, (num_labels,))

        self.assertTrue(
            torch.isclose(
                label_smoothing_loss(logits, targets, reduce=True),
                samplewise_label_smoothing_loss(logits, targets, reduce=True),
            )
        )
        self.assertTrue(
            torch.isclose(
                label_smoothing_loss(logits, targets, reduce=False),
                samplewise_label_smoothing_loss(logits, targets, reduce=False),
            ).all()
        )

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

import torch
import torch.nn.functional as F
from pytext.loss.loss import FocalLoss, BinaryFocalLoss


class FocalLossTest(unittest.TestCase):
    def test_focal_loss_base(self):

        target = torch.randint(size=(5, 1), low=0, high=10)
        score = torch.randn((5, 10))

        config = FocalLoss.Config()
        config.gamma = 0
        config.alpha = 1
        loss_fn = FocalLoss(config=config)

        val = loss_fn(
            score,
            target.reshape(
                5,
            ),
        )
        val2 = F.nll_loss(
            F.log_softmax(score, 1, dtype=torch.float32),
            target.reshape(
                5,
            ),
        )

        self.assertAlmostEqual(val.item(), val2.item())

    def test_binary_focal_loss_base(self):

        target = torch.randint(size=(5, 1), low=0, high=10)
        score = torch.randn((5, 10))

        # onehot encoded
        target_encode = torch.zeros_like(score)
        target_encode.scatter_(1, target, 1)

        config = BinaryFocalLoss.Config()
        config.gamma = 0
        config.alpha = 1
        loss_fn = BinaryFocalLoss(config=config)

        val = loss_fn(score, target_encode)
        val2 = F.binary_cross_entropy_with_logits(score, target_encode)

        self.assertAlmostEqual(val.item(), val2.item())

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

import torch
import torch.nn.functional as F
from pytext.loss.loss import CTCLoss


class CTCLossTest(unittest.TestCase):
    def test_ctc_loss(self):
        torch.manual_seed(0)

        N = 16  # Batch size
        T = 50  # Input sequence length
        C = 20  # Number of classes (including blank)
        S = 30  # Target sequence length of longest target in batch (padding length)
        S_min = 10  # Minimum target length (only for testing)

        logits = torch.randn(N, T, C)
        targets = torch.randint(1, C, (N, S), dtype=torch.long)
        input_lengths = torch.full((N,), T, dtype=torch.long)
        target_lengths = torch.randint(S_min, S, (N,), dtype=torch.long)

        config = CTCLoss.Config()
        config.blank = 0  # Needs to be set to 0 for CuDNN support.
        ctc_loss_fn = CTCLoss(config=config)

        ctc_loss_val = ctc_loss_fn(
            logits,
            targets,
            input_lengths,
            target_lengths,
        )

        # PyTorch CTC loss
        log_probs = logits.permute(1, 0, 2).log_softmax(
            2
        )  # permute to conform to CTC loss input tensor (T,N,C) in PyTorch.
        lib_ctc_loss_val = F.ctc_loss(log_probs, targets, input_lengths, target_lengths)

        self.assertAlmostEqual(ctc_loss_val.item(), lib_ctc_loss_val.item())

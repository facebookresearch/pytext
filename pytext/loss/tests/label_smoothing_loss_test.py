#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import caffe2.python.hypothesis_test_util as hu
import numpy as np
import torch
from pytext.loss.loss import LabelSmoothedCrossEntropyLoss, SourceType
from scipy.special import logsumexp


class LabelSmoothedCrossEntropyLossTest(hu.HypothesisTestCase):
    def test_label_smoothed_cross_entropy_loss_forward(self):
        torch.manual_seed(1)
        np.random.seed(1)

        for _ in range(50):
            beta = 0.1
            reduce = np.random.choice([True, False])
            num_classes = np.random.randint(1, 10)
            input_size = np.random.randint(1, 10)
            logits = torch.randn(input_size, num_classes)
            targets = torch.randint(num_classes, (input_size,))
            num_padding_indices = np.random.randint(0, int(logits.shape[0] / 2) + 1)
            padding_indices = np.random.choice(
                logits.shape[0], num_padding_indices, replace=False
            )
            targets[padding_indices] = -100

            # Only testing for None now because unexpected behavior with nll_loss.
            # Look at definition of LabelSmoothedCrossEntropyLoss for more
            # context.
            weight = None

            manual_loss = self._compute_loss_manual(
                logits,
                targets,
                weight=weight,
                reduce=reduce,
                beta=beta,
                ignore_index=-100,
            )
            loss_fn = LabelSmoothedCrossEntropyLoss(
                config=LabelSmoothedCrossEntropyLoss.Config(),
                ignore_index=-100,
                weight=weight,
            )
            label_smoothed_loss = loss_fn(logits, targets, reduce=reduce)

            diff = np.array(manual_loss) - np.array(label_smoothed_loss)
            error = np.linalg.norm(diff)
            self.assertAlmostEqual(error, 0.0, places=4)

            loss_log_config = LabelSmoothedCrossEntropyLoss.Config()
            loss_log_config.source = SourceType.LOG_PROBS
            loss_log_fn = LabelSmoothedCrossEntropyLoss(
                config=loss_log_config, ignore_index=-100, weight=weight
            )
            label_smoothed_log_loss = loss_log_fn(
                torch.nn.functional.log_softmax(logits, dim=-1), targets, reduce=reduce
            )
            log_diff = np.array(manual_loss) - np.array(label_smoothed_log_loss)
            error = np.linalg.norm(log_diff)
            self.assertAlmostEqual(error, 0.0, places=4)

    def test_label_smoothed_cross_entropy_loss_forward_source_probs(self):
        torch.manual_seed(1)
        np.random.seed(1)

        beta = 0.1
        reduce = False
        num_classes = 2
        input_size = 2
        logits = torch.Tensor([[0.5, 0.5], [0.5, 0.5]])
        targets = torch.randint(num_classes, (input_size,))
        padding_indices = 0
        targets[padding_indices] = -100

        # Only testing for None now because unexpected behavior with nll_loss.
        # Look at definition of LabelSmoothedCrossEntropyLoss for more
        # context.
        weight = None

        manual_loss = self._compute_loss_manual(
            logits, targets, weight=weight, reduce=reduce, beta=beta, ignore_index=-100
        )
        loss_fn = LabelSmoothedCrossEntropyLoss(
            config=LabelSmoothedCrossEntropyLoss.Config(source=SourceType.PROBS),
            ignore_index=-100,
            weight=weight,
        )
        label_smoothed_loss = loss_fn(logits, targets, reduce=reduce)

        diff = np.array(manual_loss) - np.array(label_smoothed_loss)
        error = np.linalg.norm(diff)
        self.assertAlmostEqual(error, 0.0, places=4)

    def _compute_negative_cross_entropy_loss(
        self, logits, targets, weight=None, reduce=True, ignore_index=-100
    ):
        batch_size = logits.shape[0]
        all_losses = []

        for i in range(batch_size):
            if targets[i] == ignore_index:
                # assign 0.0 because we don't care about this index
                loss = 0.0
            else:
                loss = logsumexp(logits[i]) - logits[i, targets[i]]
                if weight is not None:
                    loss = loss * weight[targets[i]]
            all_losses.append(loss)

        all_losses = np.array(all_losses)
        to_include_indices = targets.numpy() != ignore_index
        return np.mean(all_losses[to_include_indices]) if reduce else all_losses

    def _compute_negative_kl_divergence(
        self, logits, targets, reduce=True, ignore_index=-100
    ):
        batch_size, num_classes = logits.shape[0], logits.shape[1]
        all_kls = []

        for i in range(batch_size):
            log_probs = logits[i] - logsumexp(logits[i])
            kl_div = torch.sum(log_probs) / num_classes
            all_kls.append(kl_div)

        all_kls = np.array(all_kls)
        to_include_indices = targets.numpy() != ignore_index
        return np.mean(all_kls[to_include_indices]) if reduce else all_kls

    def _compute_loss_manual(
        self, logits, targets, ignore_index=-100, weight=None, reduce=True, beta=None
    ):
        return (1.0 - beta) * self._compute_negative_cross_entropy_loss(
            logits, targets, weight=weight, reduce=reduce, ignore_index=ignore_index
        ) - beta * self._compute_negative_kl_divergence(
            logits, targets, reduce=reduce, ignore_index=ignore_index
        )

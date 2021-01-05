#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import caffe2.python.hypothesis_test_util as hu
import numpy as np
import torch
import torch.nn.functional as F
from pytext.loss.loss import (
    LabelSmoothedCrossEntropyLoss,
    SamplewiseLabelSmoothedCrossEntropyLoss,
    SourceType,
)
from scipy.special import logsumexp


class SamplewiseLabelSmoothedCrossEntropyLossTest(hu.HypothesisTestCase):
    def test_samplewise_label_smoothed_cross_entropy_loss_forward(self):
        torch.manual_seed(1)
        np.random.seed(1)

        sources = [SourceType.LOGITS, SourceType.LOG_PROBS, SourceType.PROBS]
        for source in sources:
            for _ in range(50):
                length = np.random.randint(1, 10)
                beta = 0.1
                reduce = np.random.choice([True, False])
                num_classes = np.random.randint(1, 10)
                input_size = np.random.randint(1, 10)
                logits = torch.randn(input_size * length, num_classes)
                log_prob = F.log_softmax(logits, dim=-1)
                prob = F.softmax(logits, dim=-1)
                if source == SourceType.LOG_PROBS:
                    logits = log_prob
                elif source == SourceType.PROBS:
                    logits = prob
                targets = torch.randint(num_classes, (input_size * length,))
                num_padding_indices = np.random.randint(0, int(logits.shape[0] / 2) + 1)
                padding_indices = np.random.choice(
                    logits.shape[0], num_padding_indices, replace=False
                )
                targets[padding_indices] = -100
                ignore_index = -100

                # Only testing for None now because unexpected behavior with nll_loss.
                # Look at definition of LabelSmoothedCrossEntropyLoss for more
                # context.
                weight = None

                standard_loss_fn, standard_loss = self._compute_loss_standard(
                    logits,
                    targets,
                    weight=weight,
                    reduce=reduce,
                    beta=beta,
                    ignore_index=ignore_index,
                    source=source,
                )
                samplewise_loss_fn = SamplewiseLabelSmoothedCrossEntropyLoss(
                    config=SamplewiseLabelSmoothedCrossEntropyLoss.Config(
                        beta=beta, source=source
                    ),
                    ignore_index=ignore_index,
                    weight=weight,
                )
                samplewise_loss = samplewise_loss_fn(
                    logits, targets, reduce=reduce, batch_size=input_size
                )

                # make sure samplewise loss still computes final loss correctly
                diff = np.array(standard_loss) - np.array(samplewise_loss)
                error = np.linalg.norm(diff)
                self.assertAlmostEqual(error, 0.0, places=4)

                # now check if samplewise losses are correct
                manual_samplewise_ce = self._compute_samplewise_cross_entropy_manual(
                    logits,
                    targets,
                    ignore_index,
                    weight=weight,
                    source=source,
                    length=length,
                )
                ce_diff = np.array(
                    samplewise_loss_fn.samplewise_cross_entropy_loss
                ) - np.array(manual_samplewise_ce)
                error = np.linalg.norm(ce_diff)
                self.assertAlmostEqual(error, 0.0, places=4)

                manual_samplewise_label_smoothing = (
                    -self._compute_samplewise_negative_kl_divergence(
                        logits,
                        targets,
                        ignore_index=ignore_index,
                        length=length,
                        source=source,
                    )
                )
                label_smoothing_diff = np.array(
                    samplewise_loss_fn.samplewise_label_smoothing_loss
                ) - np.array(manual_samplewise_label_smoothing)
                error = np.linalg.norm(label_smoothing_diff)
                self.assertAlmostEqual(error, 0.0, places=4)

    def _compute_samplewise_cross_entropy_manual(
        self,
        logits,
        targets,
        ignore_index,
        weight=None,
        source=SourceType.LOGITS,
        length=1,
    ):
        samplewise_losses = []
        if source == SourceType.LOGITS:
            log_probs = F.log_softmax(logits, dim=1)
        elif source == SourceType.PROBS:
            log_probs = logits.log()
        else:
            log_probs = logits

        # assuming logits/targets are flattened when inputted, and we want to test sequences of variable length
        log_probs = log_probs.reshape(-1, log_probs.shape[1])
        targets = targets.reshape(-1, length)

        for i in range(targets.shape[0]):
            samplewise_losses.append(
                F.nll_loss(
                    log_probs[i * length : (i + 1) * length, :].reshape(
                        length, log_probs.shape[1]
                    ),
                    targets[i, :].reshape(-1),
                    ignore_index=ignore_index,
                    reduction="mean",
                    weight=weight,
                ).item()
            )
        return samplewise_losses

    def _compute_samplewise_negative_kl_divergence(
        self, logits, targets, ignore_index=-100, length=1, source=SourceType.LOGITS
    ):
        batch_size, num_classes = logits.shape[0], logits.shape[1]
        all_kls = []

        for i in range(batch_size):
            if source == SourceType.LOGITS:
                log_probs = logits[i] - logsumexp(logits[i])
            elif source == SourceType.LOG_PROBS:
                log_probs = logits[i]
            else:
                log_probs = logits[i].log()
            kl_div = torch.sum(log_probs) / num_classes
            all_kls.append(kl_div)

        all_kls = np.array(all_kls)
        all_kls = all_kls.reshape(-1, length)
        targets = targets.reshape(-1, length)
        to_include_indices = targets.numpy() != ignore_index
        lengths = to_include_indices.sum(axis=1)
        all_kls[~to_include_indices] = 0
        samplewise_kls = np.sum(all_kls, axis=1) / lengths
        samplewise_kls[np.isnan(samplewise_kls)] = 0
        return samplewise_kls

    def _compute_loss_standard(
        self,
        logits,
        targets,
        ignore_index=-100,
        weight=None,
        reduce=True,
        beta=None,
        source=SourceType.LOGITS,
    ):
        loss_fn = LabelSmoothedCrossEntropyLoss(
            config=LabelSmoothedCrossEntropyLoss.Config(beta=beta, source=source),
            ignore_index=-100,
            weight=weight,
        )
        return loss_fn, loss_fn(logits, targets, reduce=reduce)

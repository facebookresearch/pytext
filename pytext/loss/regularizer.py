#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytext.config import ConfigBase

from .loss import Loss


class Regularizer(Loss):
    """Generic regularization function to be added to a surrogate loss (e.g., cross-entropy)."""

    def __init__(self, config, ignore_index=1):
        self.ignore_index = ignore_index

    def __call__(self, logits, targets, reduce=True):
        raise NotImplementedError


class UniformRegularizer(Regularizer):
    """
    Negative KL between the uniform and predicted distribution.
        Defined as:
            - KL(U || P(Y|X)) = - sum_i U_i * log (P(Y_i | X) / U_i)
                              = - sum_i U_i * log P(Y_i|X) + H[U]
                              = - (1/n) * sum_i log P(Y_i | X) + H[U]

    H[U] does not depend on X, thus it is omitted during optimization.
    """

    def __call__(self, logits, targets, reduce=True):
        mask = targets.ne(self.ignore_index)

        loss = -logits.mean(dim=1)

        if reduce:
            return (
                loss[mask].mean()
                if mask.any()
                else torch.tensor(0.0, device=logits.device)
            )

        return loss


class EntropyRegularizer(Regularizer):
    """
    Entropy of the predicted distribution. Defined as:
        H[P(Y|X)] = - sum_i P(Y_i|X) * log P(Y_i|X)
    """

    def __call__(self, logits, targets, reduce=True):
        mask = targets.ne(self.ignore_index)

        loss = -torch.sum(logits * logits.exp(), dim=1)

        if reduce:
            return (
                loss[mask].mean()
                if mask.any()
                else torch.tensor(0.0, device=logits.device)
            )

        return loss


class AdaptiveRegularizer(Regularizer):
    """
    Adaptive variant of `UniformRegularizer` which learns the mix-in noise distribution.

    Learning Better Structured Representations using Low-Rank Adaptive Label Smoothing
    (Ghoshal+ 2021; https://openreview.net/pdf?id=5NsEIflpbSv)
    """

    class Config(ConfigBase):
        # Controls the shape of the noise distribution. Larger values of `eta` result
        # in a sharper, low-entropy distribution. Must be >= 0.
        eta: float = 0.1
        # `label_embedding_dim` and `label_embedding_dropout` control the dimension
        # and regularization, respectively, of the adaptive label embedding matrix.
        label_embedding_dim: int = 20
        label_embedding_dropout: float = 0.4

    def __init__(self, config, ignore_index=1):
        super().__init__(config, ignore_index)

        if config.eta < 0:
            raise ValueError("eta must be >= 0")
        if config.label_embedding_dropout < 0 or config.label_embedding_dropout >= 1:
            raise ValueError("label_embedding_dropout must be [0, 1)")

        self.eta = config.eta
        self.label_embedding_dim = config.label_embedding_dim
        self.label_embedding_dropout = config.label_embedding_dropout
        self.label_embedding = None

    def compute_adaptive_loss(self, logits, targets, label_embedding):
        """
        Using Equation 3 and 4, computes several terms of the adaptive penalty.
        Specifically, we implement adaptive smoothing (`smooth_term`) and
        an entropy constraint (`eta_term`).
        """

        if targets.dim() == logits.dim() - 1:
            targets = targets.unsqueeze(-1)

        U = torch.mm(
            torch.index_select(label_embedding, 0, targets.squeeze(-1)),
            label_embedding.T,
        )
        V = F.softmax(U.float(), dim=-1).to(logits.dtype)

        smooth_term = -torch.bmm(V.unsqueeze(1), logits.unsqueeze(2)).squeeze(2)
        eta_term = -self.eta * (
            -torch.bmm(U.unsqueeze(1), V.unsqueeze(2)).mean()
            + torch.logsumexp(U, axis=-1).mean()
        )
        loss = smooth_term + eta_term

        return loss

    def __call__(self, logits, targets, reduce=True):
        mask = targets.ne(self.ignore_index)

        if self.label_embedding is None:
            # Initialize label embedding matrix to ones.
            num_labels = logits.shape[1]
            self.label_embedding = nn.Parameter(
                torch.ones(num_labels, self.label_embedding_dim),
                requires_grad=True,
            ).to(device=logits.device, dtype=logits.dtype)

        loss = self.compute_adaptive_loss(
            logits,
            targets,
            F.dropout(self.label_embedding, self.label_embedding_dropout),
        )

        if reduce:
            return (
                loss[mask].mean()
                if mask.any()
                else torch.tensor(0.0, device=logits.device)
            )

        return loss

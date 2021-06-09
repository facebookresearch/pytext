#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Union

import torch
from pytext.config import ConfigBase
from pytext.config.component import create_loss

from .loss import (
    Loss,
    NLLLoss,
    HingeLoss,
    maybe_log_normalize,
    SourceType,
)
from .regularizer import UniformRegularizer, EntropyRegularizer, AdaptiveRegularizer
from .structured_loss import StructuredLoss, StructuredMarginLoss


class LabelSmoothingLoss(Loss):
    """Label loss with an optional regularizer for smoothing."""

    class Config(ConfigBase):
        beta: float = 0.1
        label_loss: Union[
            NLLLoss.Config, StructuredMarginLoss.Config, HingeLoss.Config
        ] = NLLLoss.Config()
        smoothing_loss: Union[
            UniformRegularizer.Config,
            EntropyRegularizer.Config,
            AdaptiveRegularizer.Config,
        ] = UniformRegularizer.Config()

    def __init__(self, config, ignore_index=1):
        self.beta = config.beta
        self.label_loss_fn = create_loss(config.label_loss, ignore_index=ignore_index)
        self.smoothing_loss_fn = create_loss(
            config.smoothing_loss, ignore_index=ignore_index
        )
        self.ignore_index = ignore_index

        # Tracking variables.
        self.label_loss = 0
        self.smoothing_loss = 0

    def __call__(self, logits, targets, reduce=True):
        label_loss = self.label_loss_fn(logits, targets, reduce)

        # Flatten logits if we're using a structured label loss.
        if isinstance(self.label_loss_fn, StructuredLoss):
            logits = logits.reshape(-1, logits.size(-1))
            targets = targets.view(-1)

        smoothing_loss = self.smoothing_loss_fn(logits, targets, reduce)

        # Set tracking variables.
        self.label_loss = label_loss
        self.smoothing_loss = smoothing_loss

        loss = label_loss
        if self.beta > 0:
            # if beta is 0 and logits contains -inf smoothing_loss will
            # messup
            loss += self.beta * smoothing_loss

        return loss


class SamplewiseLabelSmoothingLoss(LabelSmoothingLoss):
    """Label smoothing loss with sample-wise logging."""

    def __init__(self, config, ignore_index=-1):
        super().__init__(config, ignore_index)

        # Sample-wise tracking variables.
        self.samplewise_label_loss = 0
        self.samplewise_smoothing_loss = 0

    def _reduce_mean(
        self, logits, targets, batch_size, label_loss, smoothing_loss, reduce=True
    ):
        """
        Class-specific reduction function to extract sample-wise losses. Currently,
        passing in reduce="mean" averages over all samples without providing access
        to sample-wise losses.
        """

        # Save original losses.
        orig_label_loss = label_loss.clone()
        orig_smoothing_loss = smoothing_loss.clone()

        # Create target mask for pad tokens.
        mask = targets.ne(self.ignore_index)

        if mask.any():
            # Guarantee ignored tokens have zero contribution to loss.
            label_loss[~mask] = 0
            smoothing_loss[~mask] = 0

            # Lengths after masking.
            lengths = torch.sum(mask.reshape(batch_size, -1), dim=1)

            # Sample-wise losses (we do not consider masked tokens in this loss).
            samplewise_label_loss = (
                torch.sum(label_loss.reshape(batch_size, -1), dim=-1) / lengths
            )
            samplewise_smoothing_loss = (
                torch.sum(smoothing_loss.reshape(batch_size, -1), dim=-1) / lengths
            )

            # Replace NaNs with zero (only happens with zero length samples).
            samplewise_label_loss[torch.isnan(samplewise_label_loss)] = 0
            samplewise_smoothing_loss[torch.isnan(samplewise_smoothing_loss)] = 0

            # Update original loss to use non-masked samples.
            label_loss = label_loss[mask]
            smoothing_loss = smoothing_loss[mask]
        else:
            samplewise_label_loss = torch.zeros(batch_size, device=logits.device)
            samplewise_smoothing_loss = torch.zeros(batch_size, device=logits.device)
            label_loss = torch.zeros(mask.shape, device=logits.shape)
            smoothing_loss = torch.zeros(mask.shape, device=logits.shape)

        # If `reduce` is enabled, compute mean loss over sequence. Otherwise,
        # revert values before masking.
        label_loss = torch.mean(label_loss) if reduce else orig_label_loss
        smoothing_loss = torch.mean(smoothing_loss) if reduce else orig_smoothing_loss

        return (
            samplewise_label_loss,
            samplewise_smoothing_loss,
            label_loss,
            smoothing_loss,
        )

    def __call__(self, logits, targets, reduce=True, batch_size=None):
        label_loss = self.label_loss_fn(logits, targets, reduce=False)
        smoothing_loss = self.smoothing_loss_fn(logits, targets, reduce=False)

        # Unless specified, batch_size is equal to the length of logits.
        if batch_size is None:
            batch_size = logits.shape[0]

        # Extract sample-wise losses and reduce regular losses.
        (
            samplewise_label_loss,
            samplewise_smoothing_loss,
            label_loss,
            smoothing_loss,
        ) = self._reduce_mean(
            logits=logits,
            targets=targets,
            batch_size=batch_size,
            label_loss=label_loss,
            smoothing_loss=smoothing_loss,
            reduce=reduce,
        )

        # Set sample-wise tracking variables.
        self.samplewise_label_loss = samplewise_label_loss
        self.samplewise_smoothing_loss = samplewise_smoothing_loss
        self.samplewise_total_loss = (
            (samplewise_label_loss + self.beta * samplewise_smoothing_loss)
            if samplewise_label_loss is not None
            and samplewise_smoothing_loss is not None
            else None
        )

        # Set tracking variables.
        self.label_loss = label_loss
        self.smoothing_loss = smoothing_loss

        loss = label_loss + self.beta * smoothing_loss

        return loss


class NARSequenceLoss(Loss):
    """Joint loss over labels and length of sequences for non-autoregressive modeling."""

    class Config(ConfigBase):
        beta: float = 0.1
        assert_valid_targets: bool = True
        label_type: SourceType = SourceType.LOG_PROBS
        length_type: SourceType = SourceType.LOG_PROBS
        label_loss: LabelSmoothingLoss.Config = LabelSmoothingLoss.Config()
        length_loss: LabelSmoothingLoss.Config = LabelSmoothingLoss.Config()
        disable_label_loss: bool = False

    def __init__(self, config, ignore_index=1):
        self.beta = config.beta
        self.assert_valid_targets = config.assert_valid_targets
        self.label_type = config.label_type
        self.length_type = config.length_type

        # We can't use a structured loss for optimizing lengths.
        if isinstance(config.length_loss.label_loss, StructuredLoss):
            raise ValueError("StructuredLoss can't be used as a length loss")

        self.label_loss_fn = create_loss(config.label_loss, ignore_index=ignore_index)
        self.length_loss_fn = create_loss(config.length_loss, ignore_index=ignore_index)
        self.disable_label_loss = config.disable_label_loss

    def __call__(
        self,
        label_logits,
        label_targets,
        length_logits,
        length_targets,
        reduce=True,
    ):
        """
        label_logits: (B x T) x V_1
        label_targets: (B x T)
        length_logits: B x V_2
        length_targets: B
        """

        label_logits = maybe_log_normalize(
            logits=label_logits, logits_type=self.label_type, dim=-1
        )
        length_logits = maybe_log_normalize(
            logits=length_logits, logits_type=self.length_type, dim=-1
        )

        max_supported_dim = length_logits.size(1)
        length_targets = length_targets.unsqueeze(-1)  # (B x T) x 1

        if self.assert_valid_targets:
            if torch.any(length_targets >= max_supported_dim):
                total_violations = str(
                    length_targets[length_targets >= max_supported_dim]
                    .flatten()
                    .tolist()
                )
                raise RuntimeError(
                    f"max_supported_dim: {max_supported_dim}, "
                    f"total violations: {total_violations}"
                )
        else:
            length_targets[length_targets >= max_supported_dim] = max_supported_dim - 1

        label_loss = self.label_loss_fn(label_logits, label_targets, reduce)
        length_loss = self.length_loss_fn(
            length_logits, length_targets.squeeze(-1), reduce
        )

        if self.disable_label_loss:
            # disable the label loss and only optimize length
            # loss
            label_loss = label_loss * 0

        loss = label_loss + self.beta * length_loss

        return (
            loss,
            {
                "label_loss": label_loss,
                "length_loss": length_loss,
                "label_label_loss": self.label_loss_fn.label_loss,
                "label_smoothing_loss": self.label_loss_fn.smoothing_loss,
                "length_label_loss": self.length_loss_fn.label_loss,
                "length_smoothing_loss": self.length_loss_fn.smoothing_loss,
            },
        )


class NARSamplewiseSequenceLoss(NARSequenceLoss):
    """Non-autoregressive sequence loss with sample-wise logging."""

    class Config(NARSequenceLoss.Config):
        label_loss: SamplewiseLabelSmoothingLoss.Config = (
            SamplewiseLabelSmoothingLoss.Config()
        )
        length_loss: SamplewiseLabelSmoothingLoss.Config = (
            SamplewiseLabelSmoothingLoss.Config()
        )

    def __call__(
        self,
        label_logits,
        label_targets,
        length_logits,
        length_targets,
        reduce=True,
    ):
        """
        label_logits: (B x T) x V_1
        label_targets: (B x T)
        length_logits: B x V_2
        length_targets: B
        """

        label_logits = maybe_log_normalize(
            logits=label_logits, logits_type=self.label_type, dim=-1
        )
        length_logits = maybe_log_normalize(
            logits=length_logits, logits_type=self.length_type, dim=-1
        )

        max_length = int(torch.max(length_targets))
        batch_size = label_logits.shape[0] // max_length
        max_supported_dim = length_logits.size(1)
        length_targets = length_targets.unsqueeze(-1)  # (B x T) x 1

        if self.assert_valid_targets:
            if torch.any(length_targets >= max_supported_dim):
                total_violations = str(
                    length_targets[length_targets >= max_supported_dim]
                    .flatten()
                    .tolist()
                )
                raise RuntimeError(
                    f"max_supported_dim: {max_supported_dim}, "
                    f"total violations: {total_violations}"
                )
        else:
            length_targets[length_targets >= max_supported_dim] = max_supported_dim - 1

        label_loss = self.label_loss_fn(label_logits, label_targets, reduce, batch_size)
        length_loss = self.length_loss_fn(
            length_logits, length_targets.squeeze(-1), reduce
        )
        loss = label_loss + self.beta * length_loss

        # Log sample-wise losses.
        samplewise_losses = {
            "samplewise_label_loss": self.label_loss_fn.samplewise_total_loss,
            "samplewise_length_loss": self.length_loss_fn.samplewise_total_loss,
            "samplewise_label_label_loss": self.label_loss_fn.samplewise_label_loss,
            "samplewise_label_smoothing_loss": self.label_loss_fn.samplewise_smoothing_loss,
            "samplewise_length_label_loss": self.length_loss_fn.samplewise_label_loss,
            "samplewise_length_smoothing_loss": self.length_loss_fn.samplewise_smoothing_loss,
        }

        return (
            loss,
            {
                "label_loss": label_loss,
                "length_loss": length_loss,
                "label_label_loss": self.label_loss_fn.label_loss,
                "label_smoothing_loss": self.label_loss_fn.smoothing_loss,
                "length_label_loss": self.length_loss_fn.label_loss,
                "length_smoothing_loss": self.length_loss_fn.smoothing_loss,
                **samplewise_losses,
            },
        )

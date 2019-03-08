#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy
import torch

from .cuda import FloatTensor


def range_to_anchors_and_delta(precision_range, num_anchors):
    """Calculates anchor points from precision range.

        Args:
            precision_range: an interval (a, b), where 0.0 <= a <= b <= 1.0
            num_anchors: int, number of equally spaced anchor points.

        Returns:
            precision_values: A `Tensor` of [num_anchors] equally spaced values
                in the interval precision_range.

            delta: The spacing between the values in precision_values.

        Raises:
            ValueError: If precision_range is invalid.
    """
    # Validate precision_range.
    if len(precision_range) != 2:
        raise ValueError(
            "length of precision_range (%d) must be 2" % len(precision_range)
        )
    if not 0 <= precision_range[0] <= precision_range[1] <= 1:
        raise ValueError(
            "precision values must follow 0 <= %f <= %f <= 1"
            % (precision_range[0], precision_range[1])
        )

    # Sets precision_values uniformly between min_precision and max_precision.
    precision_values = numpy.linspace(
        start=precision_range[0], stop=precision_range[1], num=num_anchors + 1
    )[1:]

    delta = (precision_range[1] - precision_range[0]) / num_anchors
    return FloatTensor(precision_values), delta


def build_class_priors(
    labels,
    class_priors=None,
    weights=None,
    positive_pseudocount=1.0,
    negative_pseudocount=1.0,
):
    """build class priors, if necessary.
    For each class, the class priors are estimated as
    (P + sum_i w_i y_i) / (P + N + sum_i w_i),
    where y_i is the ith label, w_i is the ith weight, P is a pseudo-count of
    positive labels, and N is a pseudo-count of negative labels.

    Args:
        labels: A `Tensor` with shape [batch_size, num_classes].
            Entries should be in [0, 1].
        class_priors: None, or a floating point `Tensor` of shape [C]
            containing the prior probability of each class (i.e. the fraction of the
            training data consisting of positive examples). If None, the class
            priors are computed from `targets` with a moving average.
        weights: `Tensor` of shape broadcastable to labels, [N, 1] or [N, C],
            where `N = batch_size`, C = num_classes`
        positive_pseudocount: Number of positive labels used to initialize the class
            priors.
        negative_pseudocount: Number of negative labels used to initialize the class
            priors.
    Returns:
        class_priors: A Tensor of shape [num_classes] consisting of the
          weighted class priors, after updating with moving average ops if created.
    """
    if class_priors is not None:
        return class_priors

    N, C = labels.size()

    weighted_label_counts = (weights * labels).sum(0)

    weight_sum = weights.sum(0)

    class_priors = torch.div(
        weighted_label_counts + positive_pseudocount,
        weight_sum + positive_pseudocount + negative_pseudocount,
    )

    return class_priors


def weighted_hinge_loss(labels, logits, positive_weights=1.0, negative_weights=1.0):
    """
    Args:
        labels: one-hot representation `Tensor` of shape broadcastable to logits
        logits: A `Tensor` of shape [N, C] or [N, C, K]
        positive_weights: Scalar or Tensor
        negative_weights: same shape as positive_weights
    Returns:
        3D Tensor of shape [N, C, K], where K is length of positive weights
        or 2D Tensor of shape [N, C]
    """
    positive_weights_is_tensor = torch.is_tensor(positive_weights)
    negative_weights_is_tensor = torch.is_tensor(negative_weights)

    # Validate positive_weights and negative_weights
    if positive_weights_is_tensor ^ negative_weights_is_tensor:
        raise ValueError(
            "positive_weights and negative_weights must be same shape Tensor "
            "or both be scalars. But positive_weight_is_tensor: %r, while "
            "negative_weight_is_tensor: %r"
            % (positive_weights_is_tensor, negative_weights_is_tensor)
        )

    if positive_weights_is_tensor and (
        positive_weights.size() != negative_weights.size()
    ):
        raise ValueError(
            "shape of positive_weights and negative_weights "
            "must be the same! "
            "shape of positive_weights is {0}, "
            "but shape of negative_weights is {1}"
            % (positive_weights.size(), negative_weights.size())
        )

    # positive_term: Tensor [N, C] or [N, C, K]
    positive_term = (1 - logits).clamp(min=0) * labels
    negative_term = (1 + logits).clamp(min=0) * (1 - labels)

    if positive_weights_is_tensor and positive_term.dim() == 2:
        return (
            positive_term.unsqueeze(-1) * positive_weights
            + negative_term.unsqueeze(-1) * negative_weights
        )
    else:
        return positive_term * positive_weights + negative_term * negative_weights


def true_positives_lower_bound(labels, logits, weights):
    """
    true_positives_lower_bound defined in paper:
    "Scalable Learning of Non-Decomposable Objectives"

    Args:
        labels: A `Tensor` of shape broadcastable to logits.
        logits: A `Tensor` of shape [N, C] or [N, C, K].
            If the third dimension is present,
            the lower bound is computed on each slice [:, :, k] independently.
        weights: Per-example loss coefficients, with shape [N, 1] or [N, C]

    Returns:
        A `Tensor` of shape [C] or [C, K].
    """
    # A `Tensor` of shape [N, C] or [N, C, K]
    loss_on_positives = weighted_hinge_loss(labels, logits, negative_weights=0.0)

    weighted_loss_on_positives = (
        weights.unsqueeze(-1) * (labels - loss_on_positives)
        if loss_on_positives.dim() > weights.dim()
        else weights * (labels - loss_on_positives)
    )
    return weighted_loss_on_positives.sum(0)


def false_postives_upper_bound(labels, logits, weights):
    """
    false_positives_upper_bound defined in paper:
    "Scalable Learning of Non-Decomposable Objectives"

    Args:
        labels: A `Tensor` of shape broadcastable to logits.
        logits: A `Tensor` of shape [N, C] or [N, C, K].
            If the third dimension is present,
            the lower bound is computed on each slice [:, :, k] independently.
        weights: Per-example loss coefficients, with shape broadcast-compatible with
            that of `labels`. i.e. [N, 1] or [N, C]

    Returns:
        A `Tensor` of shape [C] or [C, K].
    """
    loss_on_negatives = weighted_hinge_loss(labels, logits, positive_weights=0)

    weighted_loss_on_negatives = (
        weights.unsqueeze(-1) * loss_on_negatives
        if loss_on_negatives.dim() > weights.dim()
        else weights * loss_on_negatives
    )
    return weighted_loss_on_negatives.sum(0)


class LagrangeMultiplier(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


def lagrange_multiplier(x):
    return LagrangeMultiplier.apply(x)

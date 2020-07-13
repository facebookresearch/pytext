#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from enum import Enum

import torch
import torch.nn.functional as F
from pytext.config import ConfigBase
from pytext.config.component import Component, ComponentType
from pytext.utils import loss as loss_utils, precision
from pytext.utils.cuda import FloatTensor
from torch import nn


class SourceType(Enum):
    LOG_PROBS = "log_probs"
    LOGITS = "logits"
    PROBS = "probs"


class Loss(Component):
    """Base class for loss functions"""

    __COMPONENT_TYPE__ = ComponentType.LOSS

    def __init__(self, config=None, *args, **kwargs):
        super().__init__(config)

    def __call__(self, logit, targets, reduce=True):
        raise NotImplementedError


class CrossEntropyLoss(Loss):
    class Config(ConfigBase):
        pass

    def __init__(self, config, ignore_index=-100, weight=None, *args, **kwargs):
        self.ignore_index = ignore_index
        self.weight = weight

    def __call__(self, logits, targets, reduce=True):
        # Don't change to F.cross_entropy() because @barlaso suggested not doing so.
        # There's some wisdom from fairseq folks that it's the preferred way.
        # Needs more testing before we can change to using F.cross_entropy().
        return F.nll_loss(
            F.log_softmax(logits, 1, dtype=torch.float32),
            targets,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction="mean" if reduce else "none",
        )


class NLLLoss(Loss):
    def __init__(self, config, ignore_index=-100, weight=None, *args, **kwargs):
        self.ignore_index = ignore_index
        self.weight = weight

    def __call__(self, log_probs, targets, reduce=True):
        return F.nll_loss(
            log_probs,
            targets,
            ignore_index=self.ignore_index,
            reduction="mean" if reduce else "none",
            weight=self.weight,
        )


class BinaryCrossEntropyLoss(Loss):
    class Config(ConfigBase):
        reweight_negative: bool = True
        reduce: bool = True

    def __call__(self, logits, targets, reduce=True):
        """
        Computes 1-vs-all binary cross entropy loss for multiclass
        classification.
        """
        # Converts targets to one-hot representation. Dim: [batch, n_classes]
        targets = (
            (
                FloatTensor(targets.size(0), logits.size(1))
                .zero_()
                .scatter_(1, targets.unsqueeze(1).data, 1)
            )
            if len(logits.size()) > 1  # If multi-class classification.
            else targets.float()
        )

        """
        `F.binary_cross_entropy` or `torch.nn.BCELoss.` requires the
        output of the previous function be already a FloatTensor.
        """
        # This weighting applies uniform class weights.
        # examples_per_class = one_hot_target.sum(0).clamp(min=1)
        # total_positive = examples_per_class.sum()
        # weights = total_positive.unsqueeze(0) / examples_per_class

        loss = F.binary_cross_entropy_with_logits(
            precision.maybe_float(logits), targets, reduction="none"
        )

        if self.config.reweight_negative:
            # This makes sure we have same weights for all negative classes and
            # single positive class. Weight is 1 for the correct class and
            # 1 / (n - 1) for other ones.
            weights = targets + (1.0 - targets) / max(1, targets.size(1) - 1.0)
            loss = loss * weights

        return loss.sum(-1).mean() if reduce else loss.sum(-1)


class CosineEmbeddingLoss(Loss):
    class Config(ConfigBase):
        margin: float = 0.0

    def __init__(self, config, *args, **kwargs):
        self.margin = config.margin

    def __call__(self, embeddings, targets, reduce=True):
        if len(embeddings) != 2:
            raise ValueError(
                f"Number of embeddings must be 2. Found {len(embeddings)} embeddings."
            )
        return F.cosine_embedding_loss(
            embeddings[0],
            embeddings[1],
            targets,
            margin=self.margin,
            reduction="mean" if reduce else "none",
        )


class MultiLabelSoftMarginLoss(Loss):
    class Config(ConfigBase):
        pass

    def __call__(self, m_out, targets, reduce=True):
        """
        Computes multi-label classification loss
        see details in torch.nn.MultiLabelSoftMarginLoss
        """

        num_classes = m_out.size()[1]
        target_labels = targets[0]

        #  each label list is padded by -1 to make every
        # observation example has the same length of list of labels
        #  since -1 is out of the index range
        # add 1 to target_labels temporarily
        tmp_target_labels = target_labels + 1

        #  the idea is similar to one_hot_targets
        #  the following encoding supports multi-label task
        #  need to delete the first-column endoing since
        #  it's for the padded label -1
        n_hot_targets = (
            FloatTensor(target_labels.size(0), num_classes + 1)
            .zero_()
            .scatter_(1, tmp_target_labels, 1)
        )[:, 1:]

        """
        `F.multilabel_soft_margin_loss` or `torch.nn.MultiLabelSoftMarginLoss.`
        requires the
        output of the previous function be already a FloatTensor.
        """

        #  default: equal weight for each class
        #  the losses are averaged over observations for each mini-batch

        loss = F.multilabel_soft_margin_loss(
            precision.maybe_float(m_out), n_hot_targets, reduction="mean"
        )

        return loss


class AUCPRHingeLoss(nn.Module, Loss):
    """area under the precision-recall curve loss,
    Reference: "Scalable Learning of Non-Decomposable Objectives", Section 5 \
    TensorFlow Implementation: \
    https://github.com/tensorflow/models/tree/master/research/global_objectives\
    """

    class Config(ConfigBase):
        """
        Attributes:
            precision_range_lower (float): the lower range of precision values over
                which to compute AUC. Must be nonnegative, `\leq precision_range_upper`,
                and `leq 1.0`.
            precision_range_upper (float): the upper range of precision values over
                which to compute AUC. Must be nonnegative, `\geq precision_range_lower`,
                and `leq 1.0`.
            num_classes (int): number of classes(aka labels)
            num_anchors (int): The number of grid points used to approximate the
                Riemann sum.
        """

        precision_range_lower: float = 0.0
        precision_range_upper: float = 1.0
        num_classes: int = 1
        num_anchors: int = 20

    def __init__(self, config, weights=None, *args, **kwargs):
        """Args:
            config: Config containing `precision_range_lower`, `precision_range_upper`,
                `num_classes`, `num_anchors`
        """
        nn.Module.__init__(self)
        Loss.__init__(self, config)

        self.num_classes = self.config.num_classes
        self.num_anchors = self.config.num_anchors
        self.precision_range = (
            self.config.precision_range_lower,
            self.config.precision_range_upper,
        )

        # Create precision anchor values and distance between anchors.
        # coresponding to [alpha_t] and [delta_t] in the paper.
        # precision_values: 1D `Tensor` of shape [K], where `K = num_anchors`
        # delta: Scalar (since we use equal distance between anchors)
        self.precision_values, self.delta = loss_utils.range_to_anchors_and_delta(
            self.precision_range, self.num_anchors
        )

        # notation is [b_k] in paper, Parameter of shape [C, K]
        # where `C = number of classes` `K = num_anchors`
        self.biases = nn.Parameter(
            FloatTensor(self.config.num_classes, self.config.num_anchors).zero_()
        )
        self.lambdas = nn.Parameter(
            FloatTensor(self.config.num_classes, self.config.num_anchors).data.fill_(
                1.0
            )
        )

    def forward(self, logits, targets, reduce=True, size_average=True, weights=None):
        """
        Args:
            logits: Variable :math:`(N, C)` where `C = number of classes`
            targets: Variable :math:`(N)` where each value is
                `0 <= targets[i] <= C-1`
            weights: Coefficients for the loss. Must be a `Tensor` of shape
                [N] or [N, C], where `N = batch_size`, `C = number of classes`.
            size_average (bool, optional): By default, the losses are averaged
                    over observations for each minibatch. However, if the field
                    sizeAverage is set to False, the losses are instead summed
                    for each minibatch. Default: ``True``
            reduce (bool, optional): By default, the losses are averaged or summed over
                observations for each minibatch depending on size_average. When reduce
                is False, returns a loss per input/target element instead and ignores
                size_average. Default: True
        """
        C = 1 if logits.dim() == 1 else logits.size(1)

        if self.num_classes != C:
            raise ValueError(
                "num classes is %d while logits width is %d" % (self.num_classes, C)
            )

        labels, weights = AUCPRHingeLoss._prepare_labels_weights(
            logits, targets, weights=weights
        )

        # Lagrange multipliers
        # Lagrange multipliers are required to be nonnegative.
        # Their gradient is reversed so that they are maximized
        # (rather than minimized) by the optimizer.
        # 1D `Tensor` of shape [K], where `K = num_anchors`
        lambdas = loss_utils.lagrange_multiplier(self.lambdas)
        # print("lambdas: {}".format(lambdas))

        # A `Tensor` of Shape [N, C, K]
        hinge_loss = loss_utils.weighted_hinge_loss(
            labels.unsqueeze(-1),
            logits.unsqueeze(-1) - self.biases,
            positive_weights=1.0 + lambdas * (1.0 - self.precision_values),
            negative_weights=lambdas * self.precision_values,
        )

        # 1D tensor of shape [C]
        class_priors = loss_utils.build_class_priors(labels, weights=weights)

        # lambda_term: Tensor[C, K]
        # according to paper, lambda_term = lambda * (1 - precision) * |Y^+|
        # where |Y^+| is number of postive examples = N * class_priors
        lambda_term = class_priors.unsqueeze(-1) * (
            lambdas * (1.0 - self.precision_values)
        )

        per_anchor_loss = weights.unsqueeze(-1) * hinge_loss - lambda_term

        # Riemann sum over anchors, and normalized by precision range
        # loss: Tensor[N, C]
        loss = per_anchor_loss.sum(2) * self.delta
        loss /= self.precision_range[1] - self.precision_range[0]

        if not reduce:
            return loss
        elif size_average:
            return loss.mean()
        else:
            return loss.sum()

    @staticmethod
    def _prepare_labels_weights(logits, targets, weights=None):
        """
        Args:
            logits: Variable :math:`(N, C)` where `C = number of classes`
            targets: Variable :math:`(N)` where each value is
                `0 <= targets[i] <= C-1`
            weights: Coefficients for the loss. Must be a `Tensor` of shape
                [N] or [N, C], where `N = batch_size`, `C = number of classes`.
        Returns:
            labels: Tensor of shape [N, C], one-hot representation
            weights: Tensor of shape broadcastable to labels
        """
        N, C = logits.size()
        # Converts targets to one-hot representation. Dim: [N, C]
        labels = FloatTensor(N, C).zero_().scatter(1, targets.unsqueeze(1).data, 1)

        if weights is None:
            weights = FloatTensor(N).data.fill_(1.0)

        if weights.dim() == 1:
            weights.unsqueeze_(-1)

        return labels, weights


class KLDivergenceBCELoss(Loss):
    class Config(ConfigBase):
        temperature: float = 1.0
        hard_weight: float = 0.0

    def __init__(self, config, ignore_index=-100, weight=None, *args, **kwargs):
        assert 0.0 <= config.hard_weight < 1.0

        self.ignore_index = ignore_index
        self.weight = weight
        self.t = config.temperature
        self.hard_weight = config.hard_weight

    def __call__(self, logits, targets, reduce=True):
        """
        Computes Kullback-Leibler divergence loss for multiclass classification
        probability distribution computed by BinaryCrossEntropyLoss loss
        """
        hard_targets, _, soft_targets_logits = targets
        # we clamp the probability between (1e-20, 1 - 1e-20) to avoid log(0) problem
        # in the calculation of KLDivergence
        soft_targets = F.sigmoid(FloatTensor(soft_targets_logits) / self.t).clamp(
            1e-20, 1 - 1e-20
        )
        probs = F.sigmoid(logits / self.t).clamp(1e-20, 1 - 1e-20)
        probs_neg = probs.neg().add(1).clamp(1e-20, 1 - 1e-20)
        soft_targets_neg = soft_targets.neg().add(1).clamp(1e-20, 1 - 1e-20)
        if self.weight is not None:
            soft_loss = (
                F.kl_div(probs.log(), soft_targets, reduction="none") * self.weight
                + F.kl_div(probs_neg.log(), soft_targets_neg, reduction="none")
                * self.weight
            )
            if reduce:
                soft_loss = soft_loss.mean()
        else:
            soft_loss = F.kl_div(
                probs.log(), soft_targets, reduction="mean" if reduce else "none"
            ) + F.kl_div(
                probs_neg.log(),
                soft_targets_neg,
                reduction="mean" if reduce else "none",
            )
        soft_loss *= self.t ** 2  # see https://arxiv.org/pdf/1503.02531.pdf

        hard_loss = 0.0
        if self.hard_weight > 0.0:
            one_hot_targets = (
                FloatTensor(hard_targets.size(0), logits.size(1))
                .zero_()
                .scatter_(1, hard_targets.unsqueeze(1).data, 1)
            )
            hard_loss = F.binary_cross_entropy_with_logits(
                logits,
                one_hot_targets,
                reduction="mean" if reduce else "none",
                weight=self.weight,
            )

        return (1.0 - self.hard_weight) * soft_loss + self.hard_weight * hard_loss


class KLDivergenceCELoss(Loss):
    class Config(ConfigBase):
        temperature: float = 1.0
        hard_weight: float = 0.0

    def __init__(self, config, ignore_index=-100, weight=None, *args, **kwargs):
        # ignore_index not easily added to kl_div loss, don't support this until needed
        assert ignore_index < 0
        assert 0.0 <= config.hard_weight < 1.0

        self.weight = weight
        self.t = config.temperature
        self.hard_weight = config.hard_weight

    def __call__(self, logits, targets, reduce=True, combine_loss=True):
        """
        Computes Kullback-Leibler divergence loss for multiclass classification
        probability distribution computed by CrossEntropyLoss loss.
        For, KL-divergence, batchmean is the right way to reduce, not just mean.
        """
        hard_targets, _, soft_targets_logits = targets
        soft_targets = F.softmax(soft_targets_logits.float() / self.t, dim=1)
        soft_targets = soft_targets.clamp(1e-10, 1 - 1e-10)
        log_probs = F.log_softmax(logits / self.t, 1)

        if self.weight is not None:
            soft_loss = (
                F.kl_div(log_probs, soft_targets, reduction="none") * self.weight
            )
            # soft_loss dim is batch_size * num_labels, while hard_loss is just
            # batch size, we have to still reduce soft_loss by the labels
            # dimension in order to be able to add the two losses.
            soft_loss = (
                torch.sum(soft_loss, dim=1).mean()
                if reduce
                else torch.sum(soft_loss, dim=1)
            )
        else:
            soft_loss = F.kl_div(
                log_probs, soft_targets, reduction="batchmean" if reduce else "none"
            )

        soft_loss *= self.t ** 2  # See https://arxiv.org/pdf/1503.02531.pdf
        hard_loss = 0.0
        if self.hard_weight > 0.0:
            hard_loss = F.cross_entropy(
                logits,
                hard_targets,
                reduction="mean" if reduce else "none",
                weight=self.weight,
            )

        return (
            (1.0 - self.hard_weight) * soft_loss + self.hard_weight * hard_loss
            if combine_loss
            else (soft_loss, hard_loss)
        )


class PairwiseRankingLoss(Loss):
    """
    Given embeddings for a query, positive response and negative response
    computes pairwise ranking hinge loss
    """

    class Config(ConfigBase):
        margin: float = 1.0

    @staticmethod
    def get_similarities(embeddings):
        pos_embed, neg_embed, query_embed = embeddings
        pos_similarity = F.cosine_similarity(query_embed, pos_embed)
        neg_similarity = F.cosine_similarity(query_embed, neg_embed)
        return pos_similarity, neg_similarity, query_embed.size(0)

    def __call__(self, logits, targets, reduce=True):
        pos_similarity, neg_similarity, batch_size = self.get_similarities(logits)
        targets_local = FloatTensor(batch_size)
        targets_local.fill_(1)  # 1: pos_similarity should be higher than neg_similarity
        return F.margin_ranking_loss(
            pos_similarity, neg_similarity, targets_local, self.config.margin
        )


class MAELoss(Loss):
    """
    Mean absolute error or L1 loss, for regression tasks.
    """

    class Config(ConfigBase):
        pass

    def __call__(self, predictions, targets, reduce=True):
        return F.l1_loss(predictions, targets, reduction="mean" if reduce else "none")


class MSELoss(Loss):
    """
    Mean squared error or L2 loss, for regression tasks.
    """

    class Config(ConfigBase):
        pass

    def __call__(self, predictions, targets, reduce=True):
        return F.mse_loss(predictions, targets, reduction="mean" if reduce else "none")


class LabelSmoothedCrossEntropyLoss(Loss):
    class Config(ConfigBase):
        beta: float = 0.1
        source: SourceType = SourceType.LOGITS
        use_entropy: bool = False

    def __init__(self, config, ignore_index=-100, weight=None, *args, **kwargs):
        # weight values other than 1.0 gives inconsistent behavior
        # Refer: https://github.com/pytorch/pytorch/issues/17577
        if weight is not None:
            assert torch.sum(torch.abs(weight - 1.0)) < 1e-7

        self.ignore_index = ignore_index
        self.weight = weight
        self.beta = config.beta
        self.source = config.source
        self.use_entropy = config.use_entropy
        self.cross_entropy_loss = None
        self.label_smoothing_loss = None

    def __call__(self, logits, targets, reduce=True):
        """
        If use_entropy is False, returns the cross-entropy loss alongwith the KL divergence of the
        discrete uniform distribution with the logits. Refer to section 3.2
        If use_entopy is True, uses the entropy of the output distribution as
        the smoothing loss (i.e., higher entropy, better). Refer to section 3
        https://arxiv.org/pdf/1701.06548.pdf
        """

        if self.use_entropy:
            # loss is negative of entropy
            probs = F.softmax(logits, dim=1)
            log_probs = torch.log(probs)
            label_smoothing_loss = torch.sum(log_probs * probs, dim=1)
        else:
            # negative KL-div has an additional log(num_classes) term but ignored
            # here because it doesn't contribute to optimization
            if self.source == SourceType.LOGITS:
                log_probs = F.log_softmax(logits, dim=1)
            elif self.source == SourceType.PROBS:
                log_probs = logits.log()
            else:
                log_probs = logits
            label_smoothing_loss = -1 * log_probs.mean(dim=1)

        if reduce:
            non_ignored = targets != self.ignore_index
            if non_ignored.any():
                label_smoothing_loss = torch.mean(label_smoothing_loss[non_ignored])
            else:
                label_smoothing_loss = torch.tensor(0.0, device=logits.device)

        cross_entropy_loss = F.nll_loss(
            log_probs,
            targets,
            ignore_index=self.ignore_index,
            reduction="mean" if reduce else "none",
            weight=self.weight,
        )

        self.cross_entropy_loss = cross_entropy_loss
        self.label_smoothing_loss = label_smoothing_loss

        return (1.0 - self.beta) * cross_entropy_loss + self.beta * label_smoothing_loss


class LabelSmoothedCrossEntropyLengthLoss(Loss):
    class Config(LabelSmoothedCrossEntropyLoss.Config):
        lengths_weight: float = 0.25
        beta_2: float = 0.25
        assert_valid_targets: bool = True

    def __init__(self, config, weight=None, ignore_index=-100):
        # weight values other than 1.0 gives inconsistent behavior
        # Refer: https://github.com/pytorch/pytorch/issues/17577
        if weight is not None:
            assert torch.sum(torch.abs(weight - 1.0)) < 1e-7

        self.lengths_weight = config.lengths_weight
        self.assert_valid_targets = config.assert_valid_targets
        self.label_smoothing_loss = LabelSmoothedCrossEntropyLoss(
            config, ignore_index=ignore_index, weight=weight
        )

        self.length_loss = LabelSmoothedCrossEntropyLoss(
            config=LabelSmoothedCrossEntropyLoss.Config(
                beta=config.beta_2,
                use_entropy=config.use_entropy,
                source=SourceType.LOG_PROBS,
            )
        )

    def __call__(self, logits, targets, length_log_probs, length_targets, reduce=True):
        label_loss = self.label_smoothing_loss(logits, targets, reduce=reduce)

        max_supported_dim = length_log_probs.size(1)
        length_targets = length_targets.unsqueeze(-1)

        if self.assert_valid_targets:
            assert not torch.any(
                length_targets >= max_supported_dim
            ), f"max_supported_dim: {max_supported_dim}, Total Violations : {str(length_targets[length_targets >= max_supported_dim].flatten().tolist())}"
        else:
            length_targets[length_targets >= max_supported_dim] = max_supported_dim - 1

        length_loss = self.length_loss(
            logits=length_log_probs, targets=length_targets.view(-1), reduce=reduce
        )

        total_loss = label_loss + self.lengths_weight * length_loss

        return (
            total_loss,
            {
                "label_loss": label_loss,
                "length_loss": length_loss,
                "labels_cross_entropy_loss": self.label_smoothing_loss.cross_entropy_loss,
                "labels_label_smoothing_loss": self.label_smoothing_loss.label_smoothing_loss,
                "lengths_cross_entropy_loss": self.length_loss.cross_entropy_loss,
                "lengths_label_smoothing_loss": self.length_loss.label_smoothing_loss,
            },
        )

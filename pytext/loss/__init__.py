#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from .loss import (
    AUCPRHingeLoss,
    BinaryCrossEntropyLoss,
    BinaryCrossEntropyWithLogitsLoss,
    CosineEmbeddingLoss,
    CrossEntropyLoss,
    KLDivergenceBCELoss,
    KLDivergenceCELoss,
    LabelSmoothedCrossEntropyLoss,
    Loss,
    MAELoss,
    MSELoss,
    HingeLoss,
    MultiLabelSoftMarginLoss,
    NLLLoss,
    PairwiseRankingLoss,
    SourceType,
    MaxMarginLoss,
)
from .regularized_loss import (
    LabelSmoothingLoss,
    SamplewiseLabelSmoothingLoss,
    NARSequenceLoss,
    NARSamplewiseSequenceLoss,
)
from .regularizer import UniformRegularizer, EntropyRegularizer, AdaptiveRegularizer


__all__ = [
    "AUCPRHingeLoss",
    "Loss",
    "CrossEntropyLoss",
    "CosineEmbeddingLoss",
    "BinaryCrossEntropyLoss",
    "BinaryCrossEntropyWithLogitsLoss",
    "HingeLoss",
    "MultiLabelSoftMarginLoss",
    "KLDivergenceBCELoss",
    "KLDivergenceCELoss",
    "MAELoss",
    "MSELoss",
    "NLLLoss",
    "PairwiseRankingLoss",
    "LabelSmoothedCrossEntropyLoss",
    "SourceType",
    "LabelSmoothingLoss",
    "SamplewiseLabelSmoothingLoss",
    "MaxMarginLoss",
    "NARSequenceLoss",
    "NARSamplewiseSequenceLoss",
    "UniformRegularizer",
    "EntropyRegularizer",
    "AdaptiveRegularizer",
]

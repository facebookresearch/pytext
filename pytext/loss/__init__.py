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
)
from .regularized_loss import (
    LabelSmoothingLoss,
    SamplewiseLabelSmoothingLoss,
    NARSequenceLoss,
    NARSamplewiseSequenceLoss,
)
from .regularizer import UniformRegularizer, EntropyRegularizer, AdaptiveRegularizer
from .structured_loss import StructuredLoss, StructuredMarginLoss, CostFunctionType


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
    "CostFunctionType",
    "StructuredLoss",
    "StructuredMarginLoss",
    "LabelSmoothingLoss",
    "SamplewiseLabelSmoothingLoss",
    "NARSequenceLoss",
    "NARSamplewiseSequenceLoss",
    "UniformRegularizer",
    "EntropyRegularizer",
    "AdaptiveRegularizer",
]

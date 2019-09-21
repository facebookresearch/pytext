#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from .loss import (
    AUCPRHingeLoss,
    BinaryCrossEntropyLoss,
    CosineEmbeddingLoss,
    CrossEntropyLoss,
    KLDivergenceBCELoss,
    KLDivergenceCELoss,
    LabelSmoothedCrossEntropyLoss,
    Loss,
    MAELoss,
    MSELoss,
    MultiLabelSoftMarginLoss,
    NLLLoss,
    PairwiseRankingLoss,
)


__all__ = [
    "AUCPRHingeLoss",
    "Loss",
    "CrossEntropyLoss",
    "CosineEmbeddingLoss",
    "BinaryCrossEntropyLoss",
    "MultiLabelSoftMarginLoss",
    "KLDivergenceBCELoss",
    "KLDivergenceCELoss",
    "MAELoss",
    "MSELoss",
    "NLLLoss",
    "PairwiseRankingLoss",
    "LabelSmoothedCrossEntropyLoss",
]

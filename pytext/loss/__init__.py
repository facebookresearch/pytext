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
    "MSELoss",
    "NLLLoss",
    "PairwiseRankingLoss",
    "LabelSmoothedCrossEntropyLoss",
]

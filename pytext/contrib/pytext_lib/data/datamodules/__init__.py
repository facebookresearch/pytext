#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .contrastive_learning import ContrastiveLearningDataModule
from .doc_classification import DocClassificationDataModule


__all__ = ["ContrastiveLearningDataModule", "DocClassificationDataModule"]

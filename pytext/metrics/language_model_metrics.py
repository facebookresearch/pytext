#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import math
from typing import NamedTuple, Optional


"""
Language model metric utilities.
"""


class LanguageModelMetric(NamedTuple):
    """
    Class for language model metrics.

    Attributes:
        perplexity_per_word: Average perplexity per word of the dataset.
    """

    perplexity_per_word: float

    def print_metrics(self):
        print(f"Perplexity per word : {self.perplexity_per_word: 0.2f}")


class LanguageModelRealtimeMetric(NamedTuple):
    """
    Language model realtime metric for tracking training progress and performance.
    Why not subclass NamedTuple: https://github.com/python/typing/issues/427.
    """

    n_batches: int
    n_updates: int
    tps: Optional[float] = None
    ppl: Optional[float] = None
    batch_ppl: Optional[float] = None
    batch_tps: Optional[float] = None


def compute_language_model_metric(loss_per_word: float) -> LanguageModelMetric:
    try:
        ppl = math.exp(loss_per_word)
    except OverflowError:
        ppl = float("inf")
    return LanguageModelMetric(perplexity_per_word=ppl)

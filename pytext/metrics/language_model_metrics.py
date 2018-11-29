#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import math
from typing import NamedTuple


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


def compute_language_model_metric(loss_per_word: float) -> LanguageModelMetric:
    return LanguageModelMetric(perplexity_per_word=math.exp(loss_per_word))

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import collections
from typing import NamedTuple


class Seq2SeqMetrics(NamedTuple):
    loss: float
    exact_match: float
    f1: float
    bleu: float

    def print_metrics(self) -> None:
        if self.exact_match is not None:
            print(f"Exact Match = {self.exact_match}")
        if self.f1 is not None:
            print(f"F1 = {self.f1}")
        if self.bleu is not None:
            print(f"BLEU = {self.bleu}")


class Seq2SeqTopKMetrics(Seq2SeqMetrics):
    k: int
    exact_match_top_k: float
    f1_top_k: float
    bleu_top_k: float

    def __new__(
        cls, loss, exact_match, f1, bleu, k, exact_match_top_k, f1_top_k, bleu_top_k
    ):
        self = super(Seq2SeqTopKMetrics, cls).__new__(cls, loss, exact_match, f1, bleu)
        self.k = k
        self.exact_match_top_k = exact_match_top_k
        self.f1_top_k = f1_top_k
        self.bleu_top_k = bleu_top_k
        return self

    def print_metrics(self) -> None:
        super().print_metrics()
        if self.exact_match_top_k is not None:
            print(
                f"Exact Match for top {self.k} predictions = {self.exact_match_top_k}"
            )
        if self.f1_top_k is not None:
            print(f"F1 for top {self.k} predictions = {self.f1_top_k}")
        if self.bleu_top_k is not None:
            print(f"BLEU for top {self.k} predictions = {self.bleu_top_k}")


def compute_f1(hypothesis_list, reference_list, eps=1e-8):
    """
    Computes token F1 given a hypothesis and reference. This is defined as
    F1 = 2 * ((P * R) / (P + R + eps)) where P = precision, R = recall, and eps
    = epsilon for smoothing zero denominators. By default, eps = 1e-8.
    """

    hypothesis_set = collections.Counter(hypothesis_list)
    reference_set = collections.Counter(reference_list)
    overlapping_set = hypothesis_set & reference_set

    hypothesis_count = len(hypothesis_list)
    reference_count = len(reference_list)
    overlapping_count = sum(overlapping_set.values())

    precision = overlapping_count / hypothesis_count if hypothesis_count > 0 else 0
    recall = overlapping_count / reference_count if reference_count > 0 else 0
    f1 = (2.0 * precision * recall) / (precision + recall + eps)

    return f1

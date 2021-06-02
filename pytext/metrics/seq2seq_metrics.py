#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import collections
from typing import NamedTuple, Dict

from pytext.metrics import (
    ClassificationMetrics,
)


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


class MaskedSeq2SeqTopKMetrics(Seq2SeqTopKMetrics):
    k: int
    exact_match_top_k: float
    f1_top_k: float
    bleu_top_k: float
    length_metrics: Dict[int, float] = None
    length_reports: ClassificationMetrics = None

    def __new__(
        cls,
        loss,
        exact_match,
        f1,
        bleu,
        k,
        exact_match_top_k,
        f1_top_k,
        bleu_top_k,
        length_metrics,
        length_reports,
    ):
        self = super(Seq2SeqTopKMetrics, cls).__new__(cls, loss, exact_match, f1, bleu)
        self.k = k
        self.exact_match_top_k = exact_match_top_k
        self.f1_top_k = f1_top_k
        self.bleu_top_k = bleu_top_k
        self.length_metrics = length_metrics
        self.length_reports = length_reports
        return self

    def print_metrics(self) -> None:
        super().print_metrics()
        if self.length_metrics:
            print("\n\nLength Metrics :", self.length_metrics)
            print(f"Length Accuracy: {self.length_reports.accuracy * 100:.2f}")
        if self.length_reports:
            print("\n\nLength Reports :", self.length_reports.print_metrics())


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

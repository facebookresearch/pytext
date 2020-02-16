#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import NamedTuple


class Seq2SeqMetrics(NamedTuple):
    accuracy: float
    loss: float
    bleu: float

    def print_metrics(self) -> None:
        if self.accuracy is not None:
            print(f"\n\nAccuracy = {self.accuracy * 100:.2f} %")
        if self.bleu is not None:
            print(f"\nBLEU = {self.bleu}")


class Seq2SeqTopKMetrics(Seq2SeqMetrics):
    k: int
    accuracy_top_k: float
    bleu_top_k: float

    def __new__(cls, accuracy, loss, bleu, k, accuracy_top_k, bleu_top_k):
        self = super(Seq2SeqTopKMetrics, cls).__new__(cls, accuracy, loss, bleu)
        self.k = k
        self.accuracy_top_k = accuracy_top_k
        self.bleu_top_k = bleu_top_k
        return self

    def print_metrics(self) -> None:
        super().print_metrics()
        if self.accuracy_top_k is not None:
            print(
                f"\n\nAccuracy for Top {self.k} predictions = "
                + f" {self.accuracy * 100:.2f} %"
            )
        if self.bleu is not None:
            print(f"\nBLEU for top {self.k} predictions = {self.bleu}")

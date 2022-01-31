#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import NamedTuple

from pytext.metrics import ClassificationMetrics


class SquadMetrics(NamedTuple):
    classification_metrics: ClassificationMetrics
    num_examples: int = 1
    exact_matches: float = -1.0
    f1_score: float = -1.0
    f1_score_pos_only: float = -1.0

    def print_metrics(self) -> None:
        print(f"Number of Examples = {self.num_examples}")
        print(f"Exact Matches = {self.exact_matches:.2f} %")
        print(f"Token Level F1 Score = {self.f1_score:.2f} %")
        print(
            f"Token Level F1 Score for positive examples = {self.f1_score_pos_only:.2f} %"
        )
        if self.classification_metrics:
            # this is NoneType if we ignore_impossible.
            print("======= Has Answer Classification Metrics =======")
            self.classification_metrics.print_metrics()

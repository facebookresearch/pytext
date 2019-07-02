#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import NamedTuple


class SquadMetrics(NamedTuple):
    num_examples: int = 1
    exact_matches: float = -1.0
    f1_score: float = -1.0

    def print_metrics(self) -> None:
        print(f"Number of samples = {self.num_examples}")
        print(f"Percentage of exact matches = {self.exact_matches}")
        print(f"F1 score = {self.f1_score}")

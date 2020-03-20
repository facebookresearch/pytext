#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import NamedTuple


class DenseRetrievalMetrics(NamedTuple):
    """
    Metric class for dense passage retrieval.

    Attributes:
        num_examples (int): number of samples
        accuracy (float): how many times did we get the +ve doc from list of docs
        average_rank (float): average rank of positive passage

    """

    num_examples: int
    accuracy: float
    average_rank: float

    def print_metrics(self) -> None:
        print(f"Number of samples = {self.num_examples}")
        print(f"Accuracy = {self.accuracy * 100:.2f}")
        print(f"Average Rank = {self.average_rank}")

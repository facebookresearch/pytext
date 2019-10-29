#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List

import torch


class VectorNormalizer(torch.nn.Module):
    """Performs in-place normalization over all features of a dense feature
    vector by doing (x - mean)/stddev for each x in the feature vector.

    This is a ScriptModule so that the normalize function can be called at
    training time in the tensorizer, as well as at inference time by using it in
    your torchscript forward function. To use this in your tensorizer
    update_meta_data must be called once per row in your initialize function,
    and then calculate_feature_stats must be called upon the last time it runs.
    See usage in FloatListTensorizer for an example.

    Setting do_normalization=False will make the normalize function an identity
    function.
    """

    def __init__(self, dim: int, do_normalization: bool = True):
        super().__init__()
        self.num_rows = 0
        self.feature_sums = [0] * dim
        self.feature_squared_sums = [0] * dim
        self.do_normalization = do_normalization
        self.feature_avgs = [0.0] * dim
        self.feature_stddevs = [1.0] * dim

    def __getstate__(self):
        return {
            "num_rows": self.num_rows,
            "feature_sums": self.feature_sums,
            "feature_squared_sums": self.feature_squared_sums,
            "do_normalization": self.do_normalization,
            "feature_avgs": self.feature_avgs,
            "feature_stddevs": self.feature_stddevs,
        }

    def __setstate__(self, state):
        self.num_rows = state["num_rows"]
        self.feature_sums = state["feature_sums"]
        self.feature_squared_sums = state["feature_squared_sums"]
        self.do_normalization = state["do_normalization"]
        self.feature_avgs = state["feature_avgs"]
        self.feature_stddevs = state["feature_stddevs"]

    # TODO: this is only to satisfy the TorchScript compiler.
    # Can remove when D17551196 lands
    def forward(self):
        pass

    def update_meta_data(self, vec):
        if self.do_normalization:
            self.num_rows += 1
            for i in range(len(vec)):
                self.feature_sums[i] += vec[i]
                self.feature_squared_sums[i] += vec[i] ** 2

    def calculate_feature_stats(self):
        if self.do_normalization:
            self.feature_avgs = [x / self.num_rows for x in self.feature_sums]
            self.feature_stddevs = [
                (
                    (self.feature_squared_sums[i] / self.num_rows)
                    - (self.feature_avgs[i] ** 2)
                )
                ** 0.5
                for i in range(len(self.feature_squared_sums))
            ]

    def normalize(self, vec: List[List[float]]):
        if self.do_normalization:
            for i in range(len(vec)):
                for j in range(len(vec[i])):
                    vec[i][j] -= self.feature_avgs[j]
                    vec[i][j] /= (
                        self.feature_stddevs[j] if self.feature_stddevs[j] != 0 else 1.0
                    )
        return vec

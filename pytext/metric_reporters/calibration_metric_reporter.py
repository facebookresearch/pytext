#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Any, Dict, List

from pytext.config import PyTextConfig
from pytext.metric_reporters.channel import Channel, ConsoleChannel
from pytext.metrics import LabelPrediction
from pytext.metrics.calibration_metrics import compute_calibration
from torch import Tensor

from .metric_reporter import MetricReporter


class CalibrationMetricReporter(MetricReporter):
    def __init__(self, channels: List[Channel], pad_index: int = -1) -> None:
        super().__init__(channels)

        self.pad_index = pad_index

    @classmethod
    def from_config(cls, config: PyTextConfig, pad_index: int = -1):
        return cls(channels=[ConsoleChannel()], pad_index=pad_index)

    def aggregate_preds(self, batch_preds: Tensor, batch_context=Dict[str, Any]):
        self.all_preds.append(batch_preds.flatten().tolist())

    def aggregate_targets(self, batch_targets: Tensor, batch_context=Dict[str, Any]):
        self.all_targets.append(batch_targets.flatten().tolist())

    def aggregate_scores(self, batch_scores: Tensor):
        batch_scores = batch_scores.view(-1, batch_scores.size(-1))
        self.all_scores.append(batch_scores.tolist())

    def calculate_metric(self):
        scores_list: List[float] = []
        preds_list: List[int] = []
        targets_list: List[int] = []

        for (scores, preds, targets) in zip(
            self.all_scores, self.all_preds, self.all_targets
        ):
            non_pad_idxs = [
                idx for (idx, target) in enumerate(targets) if target != self.pad_index
            ]

            scores = [scores[idx] for idx in non_pad_idxs]
            preds = [preds[idx] for idx in non_pad_idxs]
            targets = [targets[idx] for idx in non_pad_idxs]

            assert len(scores) == len(preds) == len(targets)

            scores_list.extend(scores)
            preds_list.extend(preds)
            targets_list.extend(targets)

        label_predictions: List[LabelPrediction] = [
            LabelPrediction(scores, pred, target)
            for (scores, pred, target) in zip(scores_list, preds_list, targets_list)
        ]

        calibration_metrics = compute_calibration(label_predictions)

        return calibration_metrics

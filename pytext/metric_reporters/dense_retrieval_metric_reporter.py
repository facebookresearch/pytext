#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from enum import Enum
from typing import Any, Dict, List

import numpy as np
from pytext.common.constants import Stage
from pytext.metric_reporters.channel import Channel, ConsoleChannel, FileChannel
from pytext.metric_reporters.metric_reporter import MetricReporter
from pytext.metrics.dense_retrieval_metrics import DenseRetrievalMetrics


class DenseRetrievalMetricNames(Enum):
    ACCURACY = "accuracy"
    AVG_RANK = "avg_rank"
    # use negative because the reporter's lower_is_better value is False
    NEGATIVE_LOSS = "negative_loss"


class DenseRetrievalMetricReporter(MetricReporter):
    class Config(MetricReporter.Config):
        text_column_names: List[str] = ["question", "positive_ctx", "negative_ctxs"]
        model_select_metric: DenseRetrievalMetricNames = (
            DenseRetrievalMetricNames.ACCURACY
        )
        # We need this because the id of positive index depens on the batch size.
        # This is needed to set the global id of the positive contexts when
        # computing average rank.
        # Set by PairwiseClassificationForDenseRetrievalTask._init_tensorizers()
        task_batch_size: int = 0
        num_negative_ctxs: int = 0

    @classmethod
    def from_config(cls, config, *args, tensorizers=None, **kwargs):
        return cls(
            channels=[ConsoleChannel(), FileChannel((Stage.TEST,), config.output_path)],
            text_column_names=config.text_column_names,
            model_select_metric=config.model_select_metric,
            task_batch_size=config.task_batch_size,
            num_negative_ctxs=config.num_negative_ctxs,
        )

    def __init__(
        self,
        channels: List[Channel],
        text_column_names: List[str],
        model_select_metric: DenseRetrievalMetricNames,
        task_batch_size: int,
        num_negative_ctxs: int = 0,
    ) -> None:
        super().__init__(channels)
        self.channels = channels
        self.text_column_names = text_column_names
        self.model_select_metric = model_select_metric

        # Assert these values to make sure that they are set explicitly.
        assert (
            task_batch_size != 0
        ), "DenseRetrievalMetricReporter: Batch size cannot be zero."
        print(f"DenseRetrievalMetricReporter: task_batch_size = {task_batch_size}")
        assert (
            num_negative_ctxs != 0
        ), "DenseRetrievalMetricReporter: Number of hard negatives cannot be zero."
        print(f"DenseRetrievalMetricReporter: num_negative_ctxs = {num_negative_ctxs}")

        self.task_batch_size = task_batch_size
        self.num_negative_ctxs = num_negative_ctxs

    def _reset(self):
        super()._reset()
        self.all_question_logits = []
        self.all_context_logits = []

    def aggregate_preds(self, preds, context):
        preds, question_logits, context_logits = preds
        super().aggregate_preds(preds)
        self.aggregate_data(self.all_question_logits, question_logits)
        self.aggregate_data(self.all_context_logits, context_logits)

    def batch_context(self, raw_batch, batch) -> Dict[str, Any]:
        context = super().batch_context(raw_batch, batch)
        for name in self.text_column_names:
            context[name] = [row[name] for row in raw_batch]
        return context

    def calculate_metric(self) -> DenseRetrievalMetrics:
        return DenseRetrievalMetrics(
            num_examples=len(self.all_preds),
            accuracy=self._get_accuracy(),
            average_rank=self._get_avg_rank(),
        )

    def get_model_select_metric(self, metrics: DenseRetrievalMetrics):
        if self.model_select_metric == DenseRetrievalMetricNames.ACCURACY:
            metric = metrics.accuracy
        elif self.model_select_metric == DenseRetrievalMetricNames.AVG_RANK:
            metric = metrics.average_rank
        else:
            raise ValueError(f"Unknown metric: {self.model_select_metric}")

        return metric

    def _get_accuracy(self):
        num_correct = sum(int(p == t) for p, t in zip(self.all_preds, self.all_targets))
        return num_correct / len(self.all_preds)

    def _get_avg_rank(self):
        dot_products = np.matmul(
            self.all_question_logits, np.transpose(self.all_context_logits)
        )
        inverse_sorted_indices = np.argsort(dot_products, axis=1)  # ascending
        positive_indices_per_question = self._get_positive_indices()

        num_questions = inverse_sorted_indices.shape[0]
        num_docs = inverse_sorted_indices.shape[1]
        rank_sum = 0
        # Sum up the rank of positive context in sorted scores for each question
        for i, pos_ctx_idx in enumerate(positive_indices_per_question):
            # Numpy returns a tuple of lists. So handle that.
            gold_idx = (inverse_sorted_indices[i] == pos_ctx_idx).nonzero()[0][0]
            rank_sum += num_docs - gold_idx

        return rank_sum / num_questions

    def _get_positive_indices(self):
        positive_indices_per_question = []
        batch_id, total_ctxs = 0, 0
        for i, pos_ctx_idx in enumerate(self.all_targets):
            if i == self.task_batch_size:
                batch_id += 1
                total_ctxs = (
                    batch_id * self.task_batch_size * (1 + self.num_negative_ctxs)
                )
            positive_indices_per_question.append(total_ctxs + pos_ctx_idx)

        return positive_indices_per_question

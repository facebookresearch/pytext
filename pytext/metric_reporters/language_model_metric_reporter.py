#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
import time

import torch
from pytext.common.constants import DatasetFieldName, Stage
from pytext.data import CommonMetadata
from pytext.metrics.language_model_metrics import (
    LanguageModelMetric,
    compute_language_model_metric,
)

from .channel import ConsoleChannel, FileChannel
from .metric_reporter import MetricReporter


class LanguageModelChannel(FileChannel):
    def get_title(self):
        return ("text", "perplexity")

    def gen_content(self, metrics, loss, preds, targets, scores, contexts):
        for i in range(len(scores)):
            yield [contexts["utterance"][i], scores[i]]


class LanguageModelMetricReporter(MetricReporter):
    UTTERANCE_COLUMN = "utterance"
    RAW_TEXT_COLUMN = "text"
    lower_is_better = True

    @classmethod
    def from_config(cls, config, meta: CommonMetadata = None, tensorizers=None):
        return cls(
            [ConsoleChannel(), LanguageModelChannel((Stage.TEST,), config.output_path)]
        )

    def add_batch_stats(
        self, n_batches, preds, targets, scores, loss, m_input, **context
    ):
        if isinstance(loss, torch.Tensor):
            loss = loss.item()
        num_words_in_batch = targets[1].sum().item()
        self.aggregate_loss += loss * num_words_in_batch
        self.total_num_tokens += num_words_in_batch

    def calculate_loss(self) -> float:
        return self.aggregate_loss / float(self.total_num_tokens)

    def _reset(self):
        super()._reset()
        self.aggregate_loss = 0.0
        self.total_num_tokens = 0

    def calculate_metric(self) -> LanguageModelMetric:
        # In language model self.total_loss is the loss per word
        return compute_language_model_metric(self.total_loss)

    def get_model_select_metric(self, metrics) -> float:
        return metrics.perplexity_per_word

    def batch_context(self, raw_batch, batch):
        context = super().batch_context(raw_batch, batch)
        if any(self.RAW_TEXT_COLUMN in row for row in raw_batch):
            context.update(
                {
                    self.UTTERANCE_COLUMN: [
                        row.get(self.RAW_TEXT_COLUMN) for row in raw_batch
                    ],
                    DatasetFieldName.TARGET_SEQ_LENS: batch["tokens"][1],
                }
            )
        return context


class MaskedLMMetricReporter(LanguageModelMetricReporter):
    @classmethod
    def from_config(cls, config, meta: CommonMetadata = None, tensorizers=None):
        return cls([ConsoleChannel()])

    def add_batch_stats(
        self, n_batches, preds, targets, scores, loss, m_input, **context
    ):
        now = time.time()

        num_words_in_batch = targets[1].sum().item()
        self.aggregate_loss += loss.item() * num_words_in_batch
        self.total_num_tokens += num_words_in_batch

        # realtime stats
        total_tokens = float(targets[2].sum())
        self.realtime_meters["tps"].update(total_tokens)

        if not n_batches % 1000:
            tps = self.realtime_meters["tps"].avg
            print(
                f"Tokens/s: {total_tokens / (now - self.time):.0f}, "
                f"batch ppl: {math.exp(loss.item()):.2f}, "
                f"agg ppl: {math.exp(self.aggregate_loss / float(self.total_num_tokens)):.2f}, "
                f"number of batches: {n_batches}, "
                f"accumulated tokens/s: {tps:.0f}",
                flush=True,
            )
        self.time = now

    def calculate_loss(self) -> float:
        return self.aggregate_loss / float(self.total_num_tokens)

    def _reset(self):
        super()._reset()
        self.aggregate_loss = 0.0
        self.total_num_tokens = 0
        self.time = time.time()

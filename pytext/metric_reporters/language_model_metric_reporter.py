#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
import time

import torch
import torch.nn.functional as F
from pytext.common.constants import Stage
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

    class Config(MetricReporter.Config):
        aggregate_metrics: bool = True

    @classmethod
    def from_config(cls, config: Config, meta: CommonMetadata = None, tensorizers=None):
        return cls(
            [ConsoleChannel(), LanguageModelChannel((Stage.TEST,), config.output_path)],
            tensorizers,
            config.aggregate_metrics,
        )

    def __init__(self, channels, tensorizers, aggregate_metrics):
        super().__init__(channels)
        self.tensorizers = tensorizers
        self.aggregate_metrics = aggregate_metrics

    def add_batch_stats(
        self, n_batches, preds, targets, scores, loss, m_input, **context
    ):
        if isinstance(loss, torch.Tensor):
            loss = loss.item()
        num_words_in_batch = targets[1].sum().item()
        self.aggregate_loss += loss * num_words_in_batch
        self.total_num_tokens += num_words_in_batch
        if self.aggregate_metrics:
            if isinstance(targets, tuple):
                targets = targets[0]
            scores = self.compute_scores(preds, targets)
            self.aggregate_scores(scores)
            self.aggregate_context(context)

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
        context = {}
        if any(self.RAW_TEXT_COLUMN in row for row in raw_batch):
            context.update(
                {
                    self.UTTERANCE_COLUMN: [
                        row.get(self.RAW_TEXT_COLUMN) for row in raw_batch
                    ]
                }
            )
        return context

    def compute_scores(self, pred, target):
        logits, pad_idx = pred
        scores = F.nll_loss(logits, target, ignore_index=pad_idx, reduction="none")
        per_sentence_loss = (torch.exp(y[y != 0].mean()) for y in scores)
        return map(lambda x: x.item(), per_sentence_loss)

    def aggregate_scores(self, scores):
        self.all_scores.extend(scores)

    def aggregate_context(self, context):
        for key, val in context.items():
            if key not in self.all_context:
                self.all_context[key] = []
            self.all_context[key].extend(val)


class MaskedLMMetricReporter(LanguageModelMetricReporter):
    @classmethod
    def from_config(cls, config, meta: CommonMetadata = None, tensorizers=None):
        return cls([ConsoleChannel()], tensorizers, config.aggregate_metrics)

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

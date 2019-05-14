#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math

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
    lower_is_better = True

    @classmethod
    def from_config(cls, config, meta: CommonMetadata = None, tensorizers=None):
        return cls(
            [ConsoleChannel(), LanguageModelChannel((Stage.TEST,), config.output_path)]
        )

    def calculate_metric(self) -> LanguageModelMetric:
        # In language model self.total_loss is the loss per word
        return compute_language_model_metric(self.total_loss)

    def _get_target_seq_lens(self):
        return self.all_context[DatasetFieldName.TARGET_SEQ_LENS]

    def calculate_loss(self) -> float:
        total_loss = n_words = pos = 0
        for loss, batch_size in zip(self.all_loss, self.batch_size):
            num_words_in_batch = sum(
                self._get_target_seq_lens()[pos : pos + batch_size]
            )
            pos = pos + batch_size
            total_loss += loss * num_words_in_batch
            n_words += num_words_in_batch
        return total_loss / float(n_words)

    def get_model_select_metric(self, metrics) -> float:
        return metrics.perplexity_per_word


class MaskedLMMetricReporter(LanguageModelMetricReporter):
    @classmethod
    def from_config(cls, config, meta: CommonMetadata = None, tensorizers=None):
        return cls([ConsoleChannel()])

    def add_batch_stats(
        self, n_batches, preds, targets, scores, loss, m_input, **context
    ):
        num_words_in_batch = targets[1].sum().item()
        self.aggregate_loss += loss.item() * num_words_in_batch
        self.total_num_tokens += num_words_in_batch

        # realtime stats
        total_tokens = float(targets[2].sum())
        self.realtime_meters["tps"].update(total_tokens)

        if not n_batches % 1000:
            tps = self.realtime_meters["tps"].avg
            print(
                f"Tokens/s: {tps:.0f}, "
                f"batch ppl: {math.exp(loss.item()):.2f}, "
                f"agg ppl: {math.exp(self.aggregate_loss / float(self.total_num_tokens)):.2f}, "
                f"number of batches: {n_batches}, "
                f"accumulated tokens/s: {tps:.0f}",
                flush=True,
            )

    def calculate_loss(self) -> float:
        return self.aggregate_loss / float(self.total_num_tokens)

    def _reset(self):
        super()._reset()
        self.aggregate_loss = 0.0
        self.total_num_tokens = 0


class MaskedLMWithClassificationMetricReporter(MaskedLMMetricReporter):
    def add_batch_stats(
        self, n_batches, preds, targets, scores, loss, m_input, **context
    ):
        loss, lm_loss, nsp_loss = loss
        num_words_in_batch = targets[1].sum().item()
        self.aggregate_loss += loss.item()
        self.aggregate_lm_loss += lm_loss.item() * num_words_in_batch
        self.total_num_tokens += num_words_in_batch
        self.aggregate_nsp_loss += nsp_loss.item()
        self.n_batches = n_batches + 1

        # realtime stats
        total_tokens = float(targets[2].sum())
        self.realtime_meters["tps"].update(total_tokens)

        if not n_batches % 1000:
            tps = self.realtime_meters["tps"].avg
            print(
                f"Tokens/s: {tps:.0f}, "
                f"BATCH ppl: {math.exp(lm_loss.item()):.2f}, "
                f"nsp_loss: {nsp_loss.item():.4f}, "
                f"lm_loss: {lm_loss.item():.4f}, "
                f"EPOCH ppl: {math.exp(self.aggregate_lm_loss / float(self.total_num_tokens)):.2f}, "
                f"nsp_loss: {self.aggregate_nsp_loss / float(self.n_batches):.4f}, "
                f"lm_loss: {self.aggregate_lm_loss / float(self.total_num_tokens):.4f}, "
                f"number of batches: {n_batches}, ",
                flush=True,
            )

    def calculate_loss(self) -> float:
        return self.aggregate_loss / float(self.n_batches)

    def _reset(self):
        super()._reset()
        self.n_batches = 1
        self.aggregate_loss = 0.0
        self.aggregate_lm_loss = 0.0
        self.aggregate_nsp_loss = 0.0
        self.total_num_tokens = 0

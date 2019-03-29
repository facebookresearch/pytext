#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
import time

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
        self.n_batches = n_batches
        self.all_loss.append(loss)
        self.batch_size.append(len(m_input[0]))
        self.aggregate_data(self.all_num_tokens, targets[1])
        now = time.time()
        if not n_batches % 1000:
            total_tokens = float(sum(targets[2]))
            print(
                f"Tokens/s: {total_tokens / (now - self.time):.0f}, ppl: {math.exp(loss):.2f}",
                flush=True,
            )
        self.time = now

    def _reset(self):
        super()._reset()
        self.all_num_tokens = []
        self.time = time.time()

    def _get_target_seq_lens(self):
        return self.all_num_tokens

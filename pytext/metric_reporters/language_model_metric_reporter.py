#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

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
    def from_config(cls, config, meta: CommonMetadata):
        return cls(
            [ConsoleChannel(), LanguageModelChannel((Stage.TEST,), config.output_path)]
        )

    def calculate_metric(self) -> LanguageModelMetric:
        # In language model self.total_loss is the loss per word
        return compute_language_model_metric(self.total_loss)

    def calculate_loss(self) -> float:
        total_loss = n_words = pos = 0
        for loss, batch_size in zip(self.all_loss, self.batch_size):
            num_words_in_batch = sum(
                self.all_context[DatasetFieldName.TARGET_SEQ_LENS][
                    pos : pos + batch_size
                ]
            )
            pos = pos + batch_size
            total_loss += loss * num_words_in_batch
            n_words += num_words_in_batch
        return total_loss / float(n_words)

    @staticmethod
    def get_model_select_metric(metrics) -> float:
        return metrics.perplexity_per_word

#!/usr/bin/env python3
import math

from pytext.data import CommonMetadata

from .channel import ConsoleChannel
from .metric_reporter import MetricReporter


class LanguageModelMetricReporter(MetricReporter):
    model_select_metric_name = "perplexity"

    @classmethod
    def from_config(cls, config, meta: CommonMetadata):
        return cls([ConsoleChannel()])

    def calculate_metric(self):
        return math.exp(self.total_loss)

    def calculate_loss(self):
        total_loss = n_words = pos = 0
        for loss, batch_size in zip(self.all_loss, self.batch_size):
            num_words_in_batch = sum(
                self.all_context["seq_lens"][pos : pos + batch_size]
            )
            pos = pos + batch_size
            total_loss += loss * num_words_in_batch
            n_words += num_words_in_batch
        return total_loss / float(n_words)

    @staticmethod
    def compare_metric(new_perplexity, old_perplexity):
        """return True if new metric indicates better model performance
        """
        if not old_perplexity:
            return True
        return new_perplexity < old_perplexity

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import json
import math
import operator
import time

import torch
import torch.nn.functional as F
from pytext.common.constants import Stage
from pytext.config.module_config import PerplexityType
from pytext.data import CommonMetadata
from pytext.metrics.language_model_metrics import (
    LanguageModelMetric,
    compute_language_model_metric,
)

from .channel import ConsoleChannel, FileChannel
from .metric_reporter import MetricReporter


PERPLEXITY_FUNC_MAP = {
    PerplexityType.MIN: torch.min,
    PerplexityType.MAX: torch.max,
    PerplexityType.MEAN: torch.mean,
    PerplexityType.MEDIAN: torch.median,
    PerplexityType.EOS: operator.itemgetter(-1),
}


def get_perplexity_func(perplexity_type):
    func = PERPLEXITY_FUNC_MAP.get(perplexity_type, None)
    if not func:
        raise NotImplementedError
    return func


class LanguageModelChannel(FileChannel):
    def get_title(self):
        return ("text", "perplexity")

    def gen_content(self, metrics, loss, preds, targets, scores, contexts):
        for i in range(len(scores)):
            yield [contexts["utterance"][i], scores[i]]


class LanguageModelMetricReporter(MetricReporter):
    UTTERANCE_COLUMN = "utterance"
    RAW_TEXT_COLUMN = "text"
    TOKENS_COLUMN = "tokens"
    lower_is_better = True

    class Config(MetricReporter.Config):
        aggregate_metrics: bool = True
        perplexity_type: PerplexityType = PerplexityType.MEDIAN

    @classmethod
    def from_config(cls, config: Config, meta: CommonMetadata = None, tensorizers=None):
        return cls(
            [ConsoleChannel(), LanguageModelChannel((Stage.TEST,), config.output_path)],
            meta,
            tensorizers,
            config.aggregate_metrics,
            config.perplexity_type,
            config.pep_format,
        )

    def __init__(
        self,
        channels,
        metadata,
        tensorizers,
        aggregate_metrics,
        perplexity_type,
        pep_format,
    ):
        super().__init__(channels, pep_format=pep_format)
        self.metadata = metadata
        self.tensorizers = tensorizers
        self.aggregate_metrics = aggregate_metrics

        assert metadata or tensorizers
        if metadata:
            self.pad_index = metadata.target.pad_token_idx
        if tensorizers:
            self.pad_index = tensorizers[self.TOKENS_COLUMN].vocab.get_pad_index()
        self.perplexity_func = get_perplexity_func(perplexity_type)

    def add_batch_stats(
        self, n_batches, preds, targets, scores, loss, m_input, **context
    ):
        if isinstance(loss, torch.Tensor):
            loss = loss.item()
        num_words_in_batch = targets[1].sum().item()
        self.aggregate_loss += loss * num_words_in_batch
        self.total_num_tokens += num_words_in_batch
        if self.aggregate_metrics:
            # unpacks logits from `targets` and computes scores for
            # each item in the batch, e.g. sentence-level perplexity
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

    def compute_scores(self, logits, targets):
        def _compute_score(tensor):
            """
            Uses a perplexity reduction function to compute a score for
            a given tensor, e.g. the mean perplexity. Filters ignored tensor
            items -- these are 0 by default.

            """

            return torch.exp(self.perplexity_func(tensor[tensor != 0.0]))

        # compute cross-entropy loss of logits wrt targets -- don't reduce
        # to access the loss of each item in the batch
        scores = F.cross_entropy(
            logits.permute(0, 2, 1),
            targets,
            ignore_index=self.pad_index,
            reduction="none",
        )
        # compute a score for each item in the batch
        return map(lambda x: _compute_score(x).item(), scores)

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
        return cls(
            [ConsoleChannel()],
            meta,
            tensorizers,
            config.aggregate_metrics,
            config.perplexity_type,
            config.pep_format,
        )

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

    def report_realtime_metric(self, stage):
        if stage != Stage.TRAIN:
            return

        if self.pep_format:
            # used for pep regression benchmark
            tps = self.realtime_meters["tps"].avg
            print(
                "PyTorchObserver "
                + json.dumps(
                    {
                        "type": "MLM",
                        "metric": "tps",
                        "unit": "token/sec",
                        "value": f"{tps:.0f}",
                    }
                ),
                flush=True,
            )

    def calculate_loss(self) -> float:
        return self.aggregate_loss / float(self.total_num_tokens)

    def _reset(self):
        super()._reset()
        self.aggregate_loss = 0.0
        self.total_num_tokens = 0
        self.time = time.time()

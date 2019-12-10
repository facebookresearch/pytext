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
from pytext.utils import cuda

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
    def get_title(self, context_keys=()):
        return ("text", "perplexity")

    def gen_content(self, metrics, loss, preds, targets, scores, contexts):
        for i in range(len(scores)):
            yield [contexts["utterance"][i], scores[i]]


class LanguageModelMetricReporter(MetricReporter):
    UTTERANCE_COLUMN = "utterance"
    RAW_TEXT_COLUMN = "text"
    TOKENS_COLUMN = "tokens"
    LABELS_COLUMN = "labels"
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
            if self.TOKENS_COLUMN in tensorizers:
                column = self.TOKENS_COLUMN
            elif self.LABELS_COLUMN in tensorizers:
                column = self.LABELS_COLUMN
            if hasattr(tensorizers[column], "vocab"):
                self.pad_index = tensorizers[column].vocab.get_pad_index()
            else:
                self.pad_index = tensorizers[column].PAD_BYTE
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

        total_masked_tokens = targets[1].sum().item()
        self.aggregate_loss += loss.item() * total_masked_tokens
        self.total_masked_tokens += total_masked_tokens

        # realtime stats
        total_tokens = float(targets[2].sum())
        self.realtime_meters["tps"].update(total_tokens)
        self.last_batch_tps = total_tokens / (now - self.time + 1e-6)
        self.last_batch_loss = loss.item()
        self.total_batches = n_batches
        self.time = now

    def report_realtime_metric(self, stage):
        if stage != Stage.TRAIN:
            return

        if cuda.DISTRIBUTED_WORLD_SIZE > 1:
            all_reduce_stats = cuda.tensor(
                [
                    self.last_batch_tps,
                    self.last_batch_loss,
                    self.aggregate_loss,
                    self.total_masked_tokens,
                    self.realtime_meters["tps"].n,
                ],
                dtype=torch.float32,
            )
            total_elapsed_time = self.realtime_meters["tps"].elapsed_time

            torch.distributed.all_reduce(all_reduce_stats)
            # average last_batch_loss by distributed_world_size
            all_reduce_stats[1:2].div_(cuda.DISTRIBUTED_WORLD_SIZE)
            [
                last_batch_tps,
                last_batch_loss,
                aggregate_loss,
                total_masked_tokens,
                total_tokens,
            ] = all_reduce_stats.tolist()
            tps = total_tokens / total_elapsed_time
        else:
            last_batch_tps = self.last_batch_tps
            last_batch_loss = self.last_batch_loss
            aggregate_loss = self.aggregate_loss
            total_masked_tokens = self.total_masked_tokens
            tps = self.realtime_meters["tps"].avg

        print(
            f"Tokens/s: {last_batch_tps:.0f}, "
            f"batch ppl: {math.exp(last_batch_loss):.2f}, "
            f"agg ppl: {math.exp(self._calculate_loss(aggregate_loss, total_masked_tokens)):.2f}, "
            f"number of batches: {self.total_batches:.0f}, "
            f"accumulated tokens/s: {tps:.0f}",
            flush=True,
        )
        # TODO: remove GPU0 report
        print(
            f"GPU-0 tokens/s: {self.last_batch_tps:.0f}, "
            f"batch ppl: {math.exp(self.last_batch_loss):.2f}, "
            f"agg ppl: {math.exp(self.calculate_loss()):.2f}, "
            f"number of batches: {self.total_batches}, "
            f"accumulated tokens/s: {self.realtime_meters['tps'].avg:.0f}",
            flush=True,
        )

        if self.pep_format:
            # used for pep regression benchmark
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
        return self._calculate_loss(self.aggregate_loss, self.total_masked_tokens)

    def _calculate_loss(self, aggregate_loss, total_masked_tokens) -> float:
        return aggregate_loss / max(1, total_masked_tokens)

    def _reset(self):
        super()._reset()
        self.aggregate_loss = 0.0
        self.total_masked_tokens = 0

    def _reset_realtime(self):
        super()._reset_realtime()
        self.last_batch_tps = 0
        self.last_batch_loss = 0
        self.total_batches = 0
        self.time = time.time()

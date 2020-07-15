#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Dict, List

import torch
from fairseq import bleu
from pytext.common.constants import (
    BatchContext,
    DatasetFieldName,
    RawExampleFieldName,
    Stage,
)
from pytext.data.tensorizers import Tensorizer
from pytext.metric_reporters.channel import ConsoleChannel, FileChannel
from pytext.metric_reporters.metric_reporter import MetricReporter
from pytext.metrics import safe_division
from pytext.metrics.seq2seq_metrics import Seq2SeqMetrics


class Seq2SeqFileChannel(FileChannel):
    def __init__(self, stages, file_path, tensorizers):
        super().__init__(stages, file_path)
        self.tensorizers = tensorizers

    def get_title(self, context_keys=()):
        return ("doc_index", "raw_input", "predictions", "targets")

    def gen_content(self, metrics, loss, preds, targets, scores, context):
        batch_size = len(targets)
        assert batch_size == len(context[DatasetFieldName.RAW_SEQUENCE]) == len(preds)
        for i in range(batch_size):
            yield [
                context[BatchContext.INDEX][i],
                context[DatasetFieldName.RAW_SEQUENCE][i],
                self.tensorizers["trg_seq_tokens"].stringify(preds[i][0]),
                self.tensorizers["trg_seq_tokens"].stringify(targets[i]),
            ]


class Seq2SeqMetricReporter(MetricReporter):
    lower_is_better = True

    class Config(MetricReporter.Config):
        pass

    def __init__(self, channels, log_gradient, tensorizers):
        super().__init__(channels, log_gradient)
        self.tensorizers = tensorizers

    def _reset(self):
        super()._reset()
        self.all_target_lens: List = []
        self.all_src_tokens: List = []

    @classmethod
    def from_config(cls, config: Config, tensorizers: Dict[str, Tensorizer]):
        return cls(
            [
                ConsoleChannel(),
                Seq2SeqFileChannel([Stage.TEST], config.output_path, tensorizers),
            ],
            tensorizers,
        )

    def add_batch_stats(
        self, n_batches, preds, targets, scores, loss, m_input, **context
    ):
        super().add_batch_stats(
            n_batches, preds, targets, scores, loss, m_input, **context
        )
        src_tokens = m_input[0]
        self.aggregate_src_tokens(src_tokens)

    def calculate_metric(self):
        num_correct = 0
        total_count = len(self.all_targets)
        trg_vocab = self.tensorizers["trg_seq_tokens"].vocab
        bleu_scorer = bleu.Scorer(
            trg_vocab.get_pad_index(),
            trg_vocab.get_eos_index(),
            trg_vocab.get_unk_index(),
        )
        for beam_pred, target in zip(self.all_preds, self.all_targets):
            pred = beam_pred[0]
            if self._compare_target_prediction_tokens(pred, target):
                num_correct = num_correct + 1
            # Bleu Metric calculation is always done with tensors on CPU or
            # type checks in fairseq/bleu.py:add() will fail
            bleu_scorer.add(torch.IntTensor(target).cpu(), torch.IntTensor(pred).cpu())

        bleu_score = 0.0 if len(self.all_preds) == 0 else bleu_scorer.score()
        accuracy = safe_division(num_correct, total_count)
        cross_entropy_loss = self.calculate_loss()
        return Seq2SeqMetrics(accuracy, cross_entropy_loss, bleu_score)

    def aggregate_targets(self, new_batch, context=None):
        if new_batch is None:
            return

        target_pad_token = self.tensorizers["trg_seq_tokens"].vocab.get_pad_index()

        self.aggregate_data(
            self.all_targets,
            [
                self._remove_tokens(targets, [target_pad_token])
                for targets in self._make_simple_list(new_batch[0])
            ],
        )
        self.aggregate_data(self.all_target_lens, new_batch[1])

    def aggregate_preds(self, new_batch, context=None):
        if new_batch is None:
            return
        self.aggregate_data(self.all_preds, new_batch)

    def aggregate_src_tokens(self, new_batch):
        src_pad_token = self.tensorizers["src_seq_tokens"].vocab.get_pad_index()

        self.aggregate_data(
            self.all_src_tokens,
            [
                self._remove_tokens(src, [src_pad_token])
                for src in self._make_simple_list(new_batch)
            ],
        )

    def gen_extra_context(self):
        self.all_context[DatasetFieldName.SOURCE_SEQ_FIELD] = self.all_src_tokens

    def _compare_target_prediction_tokens(self, prediction, target):
        return prediction == target

    def _remove_tokens(self, tokens_list, remove_tokens_list):
        cleaned_tokens = []
        for token in tokens_list:
            if isinstance(token, list):
                cleaned_tokens.append(self._remove_tokens(token, remove_tokens_list))
            elif token not in remove_tokens_list:
                cleaned_tokens.append(token)
        return cleaned_tokens

    def get_model_select_metric(self, metrics):
        return metrics.loss

    def batch_context(self, raw_batch, batch):
        return {
            DatasetFieldName.RAW_SEQUENCE: [
                row["source_sequence"] for row in raw_batch
            ],
            BatchContext.INDEX: [
                row[RawExampleFieldName.ROW_INDEX] for row in raw_batch
            ],
        }

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from collections import defaultdict
from typing import Dict, List

import numpy as np

import torch
from pytext.common.constants import BatchContext, DatasetFieldName, Stage
from pytext.data.tensorizers import Tensorizer
from pytext.metric_reporters.channel import ConsoleChannel

# These classes have been migrated to the open source directories. Imported
# here for compatibility purposes.
from pytext.metric_reporters.seq2seq_compositional import (  # noqa
    CompositionalSeq2SeqFileChannel,
    Seq2SeqCompositionalMetricReporter,
)
from pytext.metric_reporters.seq2seq_metric_reporter import (  # noqa
    Seq2SeqFileChannel,
    Seq2SeqMetricReporter,
)
from pytext.metric_reporters.seq2seq_utils import stringify
from pytext.metrics import safe_division
from pytext.metrics.mask_metrics import compute_length_metrics
from pytext.metrics.seq2seq_metrics import MaskedSeq2SeqTopKMetrics


try:
    from fairseq.scoring import bleu
except ImportError:
    from fairseq import bleu


class MaskedSeq2SeqFileChannel(Seq2SeqFileChannel):
    def get_title(self, context_keys=()):
        return (
            "doc_index",
            "raw_input",
            "tokenized_input",
            "prediction",
            "tokenized_prediction",
            "predictions_top_k",
            "targets",
        )

    def gen_content(self, metrics, loss, preds, targets, scores, context):
        batch_size = len(targets)
        assert batch_size == len(context[DatasetFieldName.RAW_SEQUENCE]) == len(preds)
        tokens_to_sentence = (
            lambda tokens_str: ("".join(tokens_str.split()))
            .replace("▁", " ")[1:]
            .lower()
        )
        for i in range(batch_size):
            yield [
                context[BatchContext.INDEX][i],
                context[DatasetFieldName.RAW_SEQUENCE][i],
                self.tensorizers["src_seq_tokens"].stringify(
                    context[DatasetFieldName.SOURCE_SEQ_FIELD][i]
                ),
                tokens_to_sentence(
                    self.tensorizers["trg_seq_tokens"].stringify(preds[i][0])
                ),
                self.tensorizers["trg_seq_tokens"].stringify(preds[i][0]),
                "|".join(
                    [
                        tokens_to_sentence(
                            self.tensorizers["trg_seq_tokens"].stringify(preds[i][j])
                        )
                        for j in range(len(preds[i]))
                    ]
                ),
                tokens_to_sentence(
                    self.tensorizers["trg_seq_tokens"].stringify(targets[i])
                ),
            ]


class MaskedSeq2SeqTopKMetricReporter(Seq2SeqMetricReporter):
    class Config(Seq2SeqMetricReporter.Config):
        model_select_metric_key: str = "all_loss"
        select_length_beam: int = 0
        log_gradient: bool = True
        TEMP_DUMP_PREDICTIONS: bool = True
        log_samplewise_losses: bool = True
        print_length_metrics: bool = True

    def __init__(
        self,
        channels,
        log_gradient,
        tensorizers,
        model_select_metric_key,
        select_length_beam,
        print_length_metrics,
    ):
        super().__init__(channels, log_gradient, tensorizers)
        self.model_select_metric_key = model_select_metric_key
        if model_select_metric_key == "em":
            self.lower_is_better = False
        else:
            self.lower_is_better = True
        self.select_length_beam = select_length_beam
        self.print_length_metrics = print_length_metrics

    @classmethod
    def from_config(cls, config: Config, tensorizers: Dict[str, Tensorizer]):
        channels = [ConsoleChannel()]
        if config.TEMP_DUMP_PREDICTIONS:
            channels.append(
                MaskedSeq2SeqFileChannel([Stage.TEST], config.output_path, tensorizers),
            )
        return cls(
            channels,
            config.log_gradient,
            tensorizers,
            config.model_select_metric_key,
            config.select_length_beam,
            config.print_length_metrics,
        )

    def _reset(self):
        super()._reset()
        self.all_target_length_preds: List = []
        self.all_beam_preds: List[List[str]] = []
        self.all_loss: Dict[str, List] = defaultdict(list)
        self.all_target_lens: List = []
        self.all_target_trees: List = []

    def add_batch_stats(
        self, n_batches, preds, targets, scores, loss, m_input, **context
    ):
        super().add_batch_stats(
            n_batches, preds, targets, scores, None, m_input, **context
        )
        self.all_loss["all_loss"].append(float(loss[0]))

        custom_losses = loss[1].keys()
        for loss_name in custom_losses:
            vals = self.all_loss[loss_name]
            # samplewise losses are stored as multi-element tensors, so need to separate cases
            if "samplewise" in loss_name:
                vals.append(loss[1][loss_name].data.cpu().numpy())
            else:
                vals.append(float(loss[1][loss_name]))

    def calculate_loss(self):
        """
        Calculate the average loss for all aggregated batch
        """
        loss_agg = {}
        for loss_name in self.all_loss.keys():
            if "samplewise" in loss_name:
                self.all_context.setdefault("losses", {})[loss_name] = np.concatenate(
                    self.all_loss[loss_name], axis=None
                )
            else:
                loss_agg[loss_name] = np.average(
                    self.all_loss[loss_name], weights=self.batch_size
                )

        return loss_agg

    def calculate_metric(self):
        total_exact_match = 0
        pred_exact_match = 0
        num_samples = len(self.all_target_trees)

        trg_vocab = self.tensorizers["trg_seq_tokens"].vocab
        bleu_scorer = bleu.Scorer(
            bleu.BleuConfig(
                pad=trg_vocab.get_pad_index(),
                eos=trg_vocab.get_eos_index(),
                unk=trg_vocab.get_unk_index(),
            )
        )

        for (beam_pred, target) in zip(self.all_beam_preds, self.all_target_trees):
            for (index, pred) in enumerate(beam_pred):
                if self._compare_target_prediction_tokens(pred, target):
                    total_exact_match += 1
                    if index == 0:
                        pred_exact_match += 1
                    break

        for (beam_preds, target) in zip(self.all_preds, self.all_targets):
            pred = beam_preds[0]
            # Bleu Metric calculation is always done with tensors on CPU or
            # type checks in fairseq/bleu.py:add() will fail
            bleu_scorer.add(torch.IntTensor(target).cpu(), torch.IntTensor(pred).cpu())

        bleu_score = round(
            0.0 if len(self.all_beam_preds) == 0 else bleu_scorer.score(), 2
        )
        exact_match = round(safe_division(pred_exact_match, num_samples) * 100.0, 2)
        exact_match_top_k = round(
            safe_division(total_exact_match, num_samples) * 100.0, 2
        )
        k = 0 if len(self.all_preds) == 0 else len(self.all_beam_preds[0])
        length_metrics, length_reports = compute_length_metrics(
            self.all_target_lens, self.all_target_length_preds, self.select_length_beam
        )
        return MaskedSeq2SeqTopKMetrics(
            loss=self.calculate_loss(),
            exact_match=exact_match,
            f1=-1,
            bleu=bleu_score,
            k=k,
            exact_match_top_k=exact_match_top_k,
            f1_top_k=-1,
            bleu_top_k=-1,
            length_metrics=length_metrics,
            length_reports=length_reports,
        )

    def aggregate_targets(self, new_batch, context=None):
        if new_batch is None:
            return
        target_vocab = self.tensorizers["trg_seq_tokens"].vocab
        target_pad_token = target_vocab.get_pad_index()
        target_bos_token = target_vocab.get_bos_index()
        target_eos_token = target_vocab.get_eos_index()

        cleaned_targets = [
            self._remove_tokens(
                target, [target_pad_token, target_eos_token, target_bos_token]
            )
            for target in self._make_simple_list(new_batch[0])
        ]

        self.aggregate_data(self.all_targets, cleaned_targets)
        self.aggregate_data(self.all_target_lens, new_batch[1])

        target_res = [stringify(target, target_vocab) for target in cleaned_targets]

        self.aggregate_data(self.all_target_trees, target_res)

    def aggregate_preds(self, new_batch, context=None):
        if new_batch is None:
            return
        tree_preds = new_batch[0]  # bsz X beam_size X seq_len
        length_preds = new_batch[1]
        target_vocab = self.tensorizers["trg_seq_tokens"].vocab
        target_pad_token = target_vocab.get_pad_index()
        target_bos_token = target_vocab.get_bos_index()
        target_eos_token = target_vocab.get_eos_index()
        cleaned_preds = [
            self._remove_tokens(
                pred, [target_pad_token, target_eos_token, target_bos_token]
            )
            for pred in self._make_simple_list(tree_preds)
        ]
        self.aggregate_data(self.all_preds, cleaned_preds)

        beam_pred_res = [
            [stringify(pred, target_vocab) for pred in beam] for beam in cleaned_preds
        ]

        self.aggregate_data(self.all_target_length_preds, length_preds)
        self.aggregate_data(self.all_beam_preds, beam_pred_res)

    def get_model_select_metric(self, metrics):
        if self.model_select_metric_key == "em":
            return metrics.exact_match
        else:
            return metrics.loss[self.model_select_metric_key]

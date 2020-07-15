#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Dict, List, Optional

from pytext.common.constants import (
    BatchContext,
    DatasetFieldName,
    RawExampleFieldName,
    Stage,
)
from pytext.data.data_structures.annotation import INVALID_TREE_STR, Annotation
from pytext.data.tensorizers import Tensorizer
from pytext.metric_reporters.channel import ConsoleChannel
from pytext.metric_reporters.compositional_metric_reporter import (
    CompositionalMetricReporter,
)
from pytext.metric_reporters.metric_reporter import MetricReporter
from pytext.metric_reporters.seq2seq_metric_reporter import (
    Seq2SeqFileChannel,
    Seq2SeqMetricReporter,
)
from pytext.metrics.intent_slot_metrics import FramePredictionPair, compute_all_metrics

from .seq2seq_utils import stringify


class CompositionalSeq2SeqFileChannel(Seq2SeqFileChannel):
    def __init__(self, stages, file_path, tensorizers, accept_flat_intents_slots):
        super().__init__(stages, file_path, tensorizers)
        self.accept_flat_intents_slots = accept_flat_intents_slots

    def get_title(self, context_keys=()):
        return (
            "row_index",
            "text",
            "predicted_output_sequence",
            "prediction",
            "target",
        )

    def validated_annotation(self, predicted_output_sequence):
        try:
            tree = Annotation(
                predicted_output_sequence,
                accept_flat_intents_slots=self.accept_flat_intents_slots,
            ).tree
        except (ValueError, IndexError):
            tree = Annotation(INVALID_TREE_STR).tree
        return tree.flat_str()

    def gen_content(self, metrics, loss, preds, targets, scores, context):
        target_vocab = self.tensorizers["trg_seq_tokens"].vocab
        batch_size = len(targets)

        for i in range(batch_size):
            yield [
                context[BatchContext.INDEX][i],
                context[DatasetFieldName.RAW_SEQUENCE][i],
                stringify(preds[i][0], target_vocab._vocab).upper(),
                self.validated_annotation(
                    stringify(preds[i][0], target_vocab._vocab).upper()
                ),
                self.validated_annotation(
                    stringify(targets[i], target_vocab._vocab).upper()
                ),
            ]


class Seq2SeqCompositionalMetricReporter(Seq2SeqMetricReporter):
    def __init__(self, channels, log_gradient, tensorizers, accept_flat_intents_slots):
        super().__init__(channels, log_gradient, tensorizers)
        self.accept_flat_intents_slots = accept_flat_intents_slots

    class Config(MetricReporter.Config):
        accept_flat_intents_slots: Optional[bool] = False

    @classmethod
    def from_config(cls, config: Config, tensorizers: Dict[str, Tensorizer]):
        return cls(
            [
                ConsoleChannel(),
                CompositionalSeq2SeqFileChannel(
                    [Stage.TEST],
                    config.output_path,
                    tensorizers,
                    config.accept_flat_intents_slots,
                ),
            ],
            tensorizers,
            config.accept_flat_intents_slots,
        )

    def _reset(self):
        super()._reset()
        self.all_target_lens: List = []
        self.all_src_tokens: List = []
        self.all_target_trees: List = []
        self.all_pred_trees: List = []

    def calculate_metric(self):
        all_metrics = compute_all_metrics(
            self.create_frame_prediction_pairs(),
            overall_metrics=True,
            calculated_loss=self.calculate_loss(),
        )
        return all_metrics

    def create_frame_prediction_pairs(self):
        return [
            FramePredictionPair(
                CompositionalMetricReporter.tree_to_metric_node(pred_tree),
                CompositionalMetricReporter.tree_to_metric_node(target_tree),
            )
            for pred_tree, target_tree in zip(
                self.all_pred_trees, self.all_target_trees
            )
        ]

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

        target_trees = [
            self.stringify_annotation_tree(target, target_vocab)
            for target in cleaned_targets
        ]

        self.aggregate_data(self.all_target_trees, target_trees)

    def aggregate_preds(self, new_batch, context=None):
        if new_batch is None:
            return

        target_vocab = self.tensorizers["trg_seq_tokens"].vocab
        target_pad_token = target_vocab.get_pad_index()
        target_bos_token = target_vocab.get_bos_index()
        target_eos_token = target_vocab.get_eos_index()

        cleaned_preds = [
            self._remove_tokens(
                pred, [target_pad_token, target_eos_token, target_bos_token]
            )
            for pred in self._make_simple_list(new_batch)
        ]

        self.aggregate_data(self.all_preds, cleaned_preds)

        pred_trees = [
            self.stringify_annotation_tree(pred[0], target_vocab)
            for pred in cleaned_preds
        ]

        self.aggregate_data(self.all_pred_trees, pred_trees)

    def stringify_annotation_tree(self, tree_tokens, tree_vocab):
        stringified_tree_str = stringify(tree_tokens, tree_vocab._vocab)
        return self.get_annotation_from_string(stringified_tree_str)

    def get_annotation_from_string(self, stringified_tree_str: str) -> Annotation:
        try:
            tree = Annotation(
                stringified_tree_str.upper(),
                accept_flat_intents_slots=self.accept_flat_intents_slots,
            ).tree
        except (ValueError, IndexError):
            tree = Annotation(INVALID_TREE_STR).tree
        return tree

    def batch_context(self, raw_batch, batch):
        return {
            DatasetFieldName.RAW_SEQUENCE: [
                row["source_sequence"] for row in raw_batch
            ],
            BatchContext.INDEX: [
                row[RawExampleFieldName.ROW_INDEX] for row in raw_batch
            ],
        }

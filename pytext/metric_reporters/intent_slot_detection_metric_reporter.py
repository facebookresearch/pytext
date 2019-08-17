#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Dict, List, Optional

from pytext.common.constants import DatasetFieldName, Stage
from pytext.data.data_structures.annotation import CLOSE, OPEN, escape_brackets
from pytext.metrics.intent_slot_metrics import (
    FramePredictionPair,
    Node,
    Span,
    compute_all_metrics,
)
from pytext.utils.data import (
    byte_length,
    get_substring_from_offsets,
    merge_token_labels_to_slot,
    parse_slot_string,
)

from .channel import Channel, ConsoleChannel, FileChannel
from .metric_reporter import MetricReporter


DOC_LABEL_NAMES = "doc_label_names"


def create_frame(text, intent_label, slot_names_str, byte_len):
    frame = Node(
        label=intent_label,
        span=Span(0, byte_len),
        children={
            Node(label=slot.label, span=Span(slot.start, slot.end))
            for slot in parse_slot_string(slot_names_str)
        },
        text=text,
    )
    return frame


def frame_to_str(frame: Node):
    annotation_str = OPEN + escape_brackets(frame.label) + " "
    cur_index = 0
    for slot in sorted(frame.children, key=lambda slot: slot.span.start):
        annotation_str += escape_brackets(
            get_substring_from_offsets(frame.text, cur_index, slot.span.start)
        )
        annotation_str += (
            OPEN
            + escape_brackets(slot.label)
            + " "
            + escape_brackets(
                get_substring_from_offsets(frame.text, slot.span.start, slot.span.end)
            )
            + " "
            + CLOSE
        )
        cur_index = slot.span.end
    annotation_str += (
        escape_brackets(get_substring_from_offsets(frame.text, cur_index, None))
        + " "
        + CLOSE
    )

    return annotation_str


class IntentSlotMetricReporter(MetricReporter):
    __EXPANSIBLE__ = True

    def __init__(
        self,
        doc_label_names: List[str],
        word_label_names: List[str],
        use_bio_labels: bool,
        channels: List[Channel],
        slot_column_name: str = "slots",
        text_column_name: str = "text",
        token_tensorizer_name: str = "tokens",
    ) -> None:
        super().__init__(channels)
        self.doc_label_names = doc_label_names
        self.word_label_names = word_label_names
        self.use_bio_labels = use_bio_labels
        self.slot_column_name = slot_column_name
        self.text_column_name = text_column_name
        self.token_tensorizer_name = token_tensorizer_name

    class Config(MetricReporter.Config):
        pass

    @classmethod
    def from_config(cls, config, tensorizers: Optional[Dict] = None):
        # TODO this part should be handled more elegantly
        for name in ["text_feats", "tokens"]:
            if name in tensorizers:
                token_tensorizer_name = name
                break
        return cls(
            tensorizers["doc_labels"].vocab,
            tensorizers["word_labels"].vocab,
            getattr(tensorizers["word_labels"], "use_bio_labels", False),
            [ConsoleChannel(), FileChannel((Stage.TEST,), config.output_path)],
            tensorizers["word_labels"].slot_column,
            tensorizers[token_tensorizer_name].text_column,
            token_tensorizer_name,
        )

    def aggregate_preds(self, batch_preds, batch_context):
        intent_preds, word_preds = batch_preds
        self.all_preds.extend(
            [
                create_frame(
                    text,
                    self.doc_label_names[intent_pred],
                    merge_token_labels_to_slot(
                        token_range[0:seq_len],
                        [self.word_label_names[p] for p in word_pred[0:seq_len]],
                        self.use_bio_labels,
                    ),
                    byte_length(text),
                )
                for text, intent_pred, word_pred, seq_len, token_range in zip(
                    batch_context[self.text_column_name],
                    intent_preds,
                    word_preds,
                    batch_context[DatasetFieldName.SEQ_LENS],
                    batch_context[DatasetFieldName.TOKEN_RANGE],
                )
            ]
        )

    def aggregate_targets(self, batch_targets, batch_context):
        intent_targets = batch_targets[0]
        self.all_targets.extend(
            [
                create_frame(
                    text,
                    self.doc_label_names[intent_target],
                    raw_slot_label,
                    byte_length(text),
                )
                for text, intent_target, raw_slot_label, seq_len in zip(
                    batch_context[self.text_column_name],
                    intent_targets,
                    batch_context[DatasetFieldName.RAW_WORD_LABEL],
                    batch_context[DatasetFieldName.SEQ_LENS],
                )
            ]
        )

    def get_raw_slot_str(self, raw_data_row):
        return ",".join([str(x) for x in raw_data_row[self.slot_column_name]])

    def aggregate_scores(self, batch_scores):
        intent_scores, slot_scores = batch_scores
        self.all_scores.extend(
            (intent_score, slot_score)
            for intent_score, slot_score in zip(
                intent_scores.tolist(), slot_scores.tolist()
            )
        )

    def predictions_to_report(self):
        """
        Generate human readable predictions
        """
        return [frame_to_str(frame) for frame in self.all_preds]

    def targets_to_report(self):
        """
        Generate human readable targets
        """
        return [frame_to_str(frame) for frame in self.all_targets]

    def calculate_metric(self):
        return compute_all_metrics(
            [
                FramePredictionPair(pred_frame, target_frame)
                for pred_frame, target_frame in zip(self.all_preds, self.all_targets)
            ],
            frame_accuracy=True,
        )

    def batch_context(self, raw_batch, batch):
        context = super().batch_context(raw_batch, batch)
        context[self.text_column_name] = [
            row[self.text_column_name] for row in raw_batch
        ]
        context[DatasetFieldName.SEQ_LENS] = batch[self.token_tensorizer_name][
            1
        ].tolist()
        context[DatasetFieldName.TOKEN_RANGE] = batch[self.token_tensorizer_name][
            2
        ].tolist()
        context[DatasetFieldName.RAW_WORD_LABEL] = [
            self.get_raw_slot_str(raw_data_row) for raw_data_row in raw_batch
        ]
        return context

    def get_model_select_metric(self, metrics):
        return metrics.frame_accuracy

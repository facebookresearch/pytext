#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Dict, List, Tuple

from pytext.common.constants import BatchContext, DatasetFieldName, Stage
from pytext.data import CommonMetadata
from pytext.data.data_structures.annotation import CLOSE, OPEN, escape_brackets
from pytext.metrics.intent_slot_metrics import (
    FramePredictionPair,
    Node,
    Span,
    compute_all_metrics,
)
from pytext.utils.data_utils import parse_slot_string
from pytext.utils.test_utils import merge_token_labels_to_slot

from .channel import Channel, ConsoleChannel, FileChannel
from .metric_reporter import MetricReporter


DOC_LABEL_NAMES = "doc_label_names"


class IntentSlotChannel(FileChannel):
    def get_title(self):
        return ("doc_index", "text", "predicted_annotation", "actual_annotation")

    def gen_content(self, metrics, loss, preds, targets, scores, context):
        for (
            index,
            utterance,
            intent_pred,
            intent_target,
            slots_pred_label,
            slots_target_label,
        ) in zip(
            context[BatchContext.INDEX],
            context[DatasetFieldName.UTTERANCE_FIELD],
            preds[0],
            targets[0],
            context["slots_prediction"],
            context[DatasetFieldName.RAW_WORD_LABEL],
        ):
            yield (
                index,
                utterance,
                self.create_annotation(
                    utterance, context[DOC_LABEL_NAMES][intent_pred], slots_pred_label
                ),
                self.create_annotation(
                    utterance,
                    context[DOC_LABEL_NAMES][intent_target],
                    slots_target_label,
                ),
            )

    @staticmethod
    def create_annotation(utterance: str, intent_label: str, slots_label: str) -> str:
        annotation_str = OPEN + escape_brackets(intent_label) + " "
        slots = parse_slot_string(slots_label)
        cur_index = 0
        for slot in slots:
            annotation_str += escape_brackets(utterance[cur_index : slot.start])
            annotation_str += (
                OPEN
                + escape_brackets(slot.label)
                + " "
                + escape_brackets(utterance[slot.start : slot.end])
                + " "
                + CLOSE
            )
            cur_index = slot.end
        annotation_str += escape_brackets(utterance[cur_index:]) + " " + CLOSE

        return annotation_str


def create_frame(intent_label, slot_names_str, utterance):
    frame = Node(
        label=intent_label,
        span=Span(0, len(utterance)),
        children={
            Node(label=slot.label, span=Span(slot.start, slot.end))
            for slot in parse_slot_string(slot_names_str)
        },
    )
    return frame


class IntentSlotMetricReporter(MetricReporter):
    def __init__(
        self,
        doc_label_names: List[str],
        word_label_names: List[str],
        use_bio_labels: bool,
        channels: List[Channel],
    ) -> None:
        super().__init__(channels)
        self.doc_label_names = doc_label_names
        self.word_label_names = word_label_names
        self.use_bio_labels = use_bio_labels

    @classmethod
    def from_config(cls, config, meta: CommonMetadata):
        doc_label_meta, word_label_meta = meta.target
        return cls(
            doc_label_meta.vocab.itos,
            word_label_meta.vocab.itos,
            word_label_meta.use_bio_labels,
            [ConsoleChannel(), IntentSlotChannel((Stage.TEST,), config.output_path)],
        )

    def _reset(self):
        self.all_doc_preds: List = []
        self.all_doc_targets: List = []
        self.all_doc_scores: List = []
        self.all_word_preds: List = []
        self.all_word_targets: List = []
        self.all_word_scores: List = []

        self.all_preds: Tuple = (self.all_doc_preds, self.all_word_preds)
        self.all_targets: Tuple = (self.all_doc_targets, self.all_word_targets)
        self.all_scores: Tuple = (self.all_doc_scores, self.all_word_scores)
        self.all_context: Dict = {}
        self.all_loss: List = []
        self.n_batches = 0
        self.batch_size: List = []

    def aggregate_preds(self, new_batch):
        self.aggregate_data(self.all_doc_preds, new_batch[0])
        self.aggregate_data(self.all_word_preds, new_batch[1])

    def aggregate_targets(self, new_batch):
        self.aggregate_data(self.all_doc_targets, new_batch[0])
        self.aggregate_data(self.all_word_targets, new_batch[1])

    def aggregate_scores(self, new_batch):
        self.aggregate_data(self.all_doc_scores, new_batch[0])
        self.aggregate_data(self.all_word_scores, new_batch[1])

    def process_pred(self, pred: List[int]) -> List[str]:
        """pred is a list of token label index
        """
        return [self.word_label_names[p] for p in pred]

    def gen_extra_context(self):
        self.all_context["slots_prediction"] = [
            merge_token_labels_to_slot(
                token_range,
                self.process_pred(word_pred[0:seq_len]),
                self.use_bio_labels,
            )
            for word_pred, seq_len, token_range in zip(
                self.all_word_preds,
                self.all_context[DatasetFieldName.SEQ_LENS],
                self.all_context[DatasetFieldName.TOKEN_RANGE],
            )
        ]
        self.all_context[DOC_LABEL_NAMES] = self.doc_label_names

    def calculate_metric(self):
        return compute_all_metrics(
            [
                FramePredictionPair(
                    create_frame(
                        self.doc_label_names[intent_pred], slots_pred, utterance
                    ),
                    create_frame(
                        self.doc_label_names[intent_target], slots_label, utterance
                    ),
                )
                for intent_pred, intent_target, slots_pred, slots_label, utterance in zip(
                    self.all_doc_preds,
                    self.all_doc_targets,
                    self.all_context["slots_prediction"],
                    self.all_context[DatasetFieldName.RAW_WORD_LABEL],
                    self.all_context[DatasetFieldName.UTTERANCE_FIELD],
                )
            ],
            frame_accuracy=True,
        )

    @staticmethod
    def get_model_select_metric(metrics):
        return metrics.frame_accuracy

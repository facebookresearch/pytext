#!/usr/bin/env python3

from typing import Dict, List, Tuple

from pytext.common.constants import DatasetFieldName, Stage
from pytext.data import CommonMetadata
from pytext.metrics import (
    FramePredictionPair,
    Node,
    Span,
    compute_all_metrics,
)
from pytext.utils.data_utils import parse_slot_string
from pytext.utils.test_utils import merge_token_labels_to_slot

from .channel import Channel, ConsoleChannel, FileChannel
from .metric_reporter import MetricReporter
from .word_tagging_metric_reporter import strip_bio_prefix


class IntentSlotChannel(FileChannel):
    def get_title(self):
        return (
            "doc_index",
            "intent_prediction",
            "intent_label",
            "slots_prediction",
            "slots_label",
            "text",
        )

    def gen_content(self, metrics, loss, preds, targets, scores, context):
        doc_preds = preds[0]
        doc_targets = targets[0]

        for i in range(len(doc_preds)):
            yield [
                context[DatasetFieldName.INDEX_FIELD][i],
                doc_preds[i],
                doc_targets[i],
                context["slots_prediction"][i],
                context[DatasetFieldName.RAW_WORD_LABEL][i],
                context[DatasetFieldName.UTTERANCE_FIELD][i],
            ]


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
    model_select_metric_name = "frame_accuracy"

    def __init__(
        self,
        doc_label_names: List[str],
        word_label_names: List[str],
        channels: List[Channel],
    ) -> None:
        super().__init__(channels)
        self.doc_label_names = doc_label_names
        self.word_label_names = word_label_names

    @classmethod
    def from_config(cls, config, meta: CommonMetadata):
        doc_label_names, word_label_names = [
            label.vocab.itos for label in meta.labels.values()
        ]
        return cls(
            doc_label_names,
            word_label_names,
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
        return [strip_bio_prefix(self.word_label_names[p]) for p in pred]

    def gen_extra_context(self):
        self.all_context["slots_prediction"] = [
            merge_token_labels_to_slot(
                token_range, self.process_pred(word_pred[0:seq_len])
            )
            for word_pred, seq_len, token_range in zip(
                self.all_word_preds,
                self.all_context[DatasetFieldName.SEQ_LENS],
                self.all_context[DatasetFieldName.TOKEN_RANGE_PAIR],
            )
        ]

    def get_meta(self):
        return {
            "doc_label_names": self.doc_label_names,
            "word_label_names": self.word_label_names,
        }

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

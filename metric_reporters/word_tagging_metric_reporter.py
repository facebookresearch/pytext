#!/usr/bin/env python3
from collections import Counter
from typing import List

from pytext.common.constants import DatasetFieldName, Stage
from pytext.data import CommonMetadata
from pytext.metrics import (
    Node,
    NodesPredictionPair,
    Span,
    compute_classification_metrics_from_nodes_pairs,
)
from pytext.utils.data_utils import Slot, parse_slot_string
from pytext.utils.test_utils import merge_token_labels_to_slot

from .channel import Channel, ConsoleChannel, FileChannel
from .metric_reporter import MetricReporter


def strip_bio_prefix(label):
    if label.startswith(Slot.B_LABEL_PREFIX) or label.startswith(Slot.I_LABEL_PREFIX):
        label = label[len(Slot.B_LABEL_PREFIX) :]
    return label


def get_slots(word_names):
    slots = {
        Node(label=slot.label, span=Span(slot.start, slot.end))
        for slot in parse_slot_string(word_names)
    }
    return Counter(slots)


class WordTaggingMetricReporter(MetricReporter):
    model_select_metric_name = "f1"

    def __init__(self, label_names: List[str], channels: List[Channel]) -> None:
        super().__init__(channels)
        self.label_names = label_names

    @classmethod
    def from_config(cls, config, meta: CommonMetadata):
        label_names = next(iter(meta.labels.values())).vocab.itos
        return cls(
            label_names,
            [ConsoleChannel(), FileChannel((Stage.TEST,), config.output_path)],
        )

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

    def process_pred(self, pred: List[int]) -> List[str]:
        """pred is a list of token label index
        """
        return [strip_bio_prefix(self.label_names[p]) for p in pred]

    def calculate_metric(self):
        return compute_classification_metrics_from_nodes_pairs(
            [
                NodesPredictionPair(
                    get_slots(
                        merge_token_labels_to_slot(
                            token_range, self.process_pred(pred[0:seq_len])
                        )
                    ),
                    get_slots(slots_label),
                )
                for pred, seq_len, token_range, slots_label in zip(
                    self.all_preds,
                    self.all_context[DatasetFieldName.SEQ_LENS],
                    self.all_context[DatasetFieldName.TOKEN_RANGE_PAIR],
                    self.all_context[DatasetFieldName.RAW_WORD_LABEL],
                )
            ]
        )[1]

    @staticmethod
    def get_model_select_metric(metrics):
        return metrics.micro_scores.f1

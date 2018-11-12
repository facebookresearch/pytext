#!/usr/bin/env python3
from typing import Sequence

from pytext.utils.data_utils import Slot


class ResultRow:
    def __init__(self, name, metrics_dict):
        self.name = name
        for m_name, m_val in metrics_dict.items():
            setattr(self, m_name, m_val)


class ResultTable:
    def __init__(self, metrics, class_names, labels, preds):
        self.rows = []
        for i, class_n in enumerate(class_names):
            metrics_dict = {}
            metrics_dict["num_samples"] = int(metrics[3][i])
            metrics_dict["num_correct"] = sum(
                int(label) == i and int(label) == int(preds[j])
                for j, label in enumerate(labels)
            )
            metrics_dict["precision"] = metrics[0][i]
            metrics_dict["recall"] = metrics[1][i]
            metrics_dict["f1"] = metrics[2][i]
            self.rows.append(ResultRow(class_n, metrics_dict))


def strip_bio_prefix(label):
    if label.startswith((Slot.B_LABEL_PREFIX, Slot.I_LABEL_PREFIX)):
        label = label[len(Slot.B_LABEL_PREFIX):]
    return label


def merge_token_bio_labels_to_slots(token_ranges, labels):
    begin, end = None, None
    prev_label = None
    for span, label in zip(token_ranges, labels):
        # Take action only if the prefix is not i
        if not label.startswith(Slot.I_LABEL_PREFIX):
            if begin is not None:
                # Label the previous chunk
                yield Slot(strip_bio_prefix(prev_label), begin, end)
                begin = None
            if label.startswith(Slot.B_LABEL_PREFIX):
                # Assign the begin location of new chunk
                begin, _ = span
        _, end = span
        prev_label = label

    # Take last token into account
    if begin is not None:
        _, end = token_ranges[-1]
        yield Slot(strip_bio_prefix(labels[-1]), begin, end)


def merge_adjacent_token_labels_to_slots(token_ranges, labels):
    begin, end = token_ranges[0]
    for prev_label, label, span in zip(labels, labels[1:], token_ranges[1:]):
        if label == prev_label:
            # Extend span
            _, end = span
        elif prev_label == Slot.NO_LABEL_SLOT:
            # Update and skip
            begin, end = span
        else:
            # Update and start new
            yield Slot(prev_label, begin, end)
            begin, end = span

    # Take last token into account
    if labels[-1] != Slot.NO_LABEL_SLOT:
        yield Slot(labels[-1], begin, end)


def merge_token_labels_to_slots(token_ranges, labels, use_bio_labels=True):
    if use_bio_labels:
        return merge_token_bio_labels_to_slots(token_ranges, labels)
    return merge_adjacent_token_labels_to_slots(token_ranges, labels)


def format_token_labels(token_labels: Sequence[Slot]) -> str:
    return ",".join(repr(slot) for slot in token_labels)

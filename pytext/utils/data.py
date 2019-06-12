#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import json
import unicodedata
from typing import Any, List, Tuple


def simple_tokenize(s: str) -> List[str]:
    return s.split(" ")


def no_tokenize(s: Any) -> Any:
    return s


class Slot:
    B_LABEL_PREFIX = "B-"
    I_LABEL_PREFIX = "I-"
    NO_LABEL_SLOT = "NoLabel"

    def __init__(self, label: str, start: int, end: int) -> None:
        self.label = label
        self.start = start
        self.end = end

    def token_overlap(self, token_start, token_end):
        start = min(token_end, max(token_start, self.start))
        end = min(token_end, max(token_start, self.end))
        return end - start

    def token_label(self, use_bio_labels, token_start, token_end):
        token_label = self.NO_LABEL_SLOT
        token_overlap = self.token_overlap(token_start, token_end)

        if use_bio_labels:
            if token_start == self.start and token_overlap:
                token_label = self.b_label_name
            elif token_start > self.start and token_overlap:
                token_label = self.i_label_name
        else:
            if token_overlap:
                token_label = self.label
        return token_label

    @property
    def b_label_name(self):
        return "{}{}".format(self.B_LABEL_PREFIX, self.label)

    @property
    def i_label_name(self):
        return "{}{}".format(self.I_LABEL_PREFIX, self.label)

    def __repr__(self):
        return "{}:{}:{}".format(self.start, self.end, self.label)


def parse_slot_string(slots_field: str) -> List[Slot]:
    slots = slots_field.split(",")
    slot_list = []
    for slot in slots:
        slot_toks = slot.split(":", 2)
        if len(slot_toks) == 3:
            curr_slot = Slot(slot_toks[2], int(slot_toks[0]), int(slot_toks[1]))
            slot_list.append(curr_slot)
    return slot_list


def parse_token(
    utterance: str, token_range: List[int]
) -> List[Tuple[str, Tuple[int, int]]]:
    range_bounds = [
        (token_range[i], token_range[i + 1]) for i in range(0, len(token_range) - 1, 2)
    ]
    return [(utterance[s:e], (s, e)) for (s, e) in range_bounds]


def parse_json_array(json_text: str) -> List[str]:
    return json.loads(json_text)


# In order to process each field independently, we need to align slot labels
def align_slot_labels(
    token_ranges: List[Tuple[int, int]], slots_field: str, use_bio_labels: bool = False
):
    return " ".join(
        parse_and_align_slot_labels_list(token_ranges, slots_field, use_bio_labels)
    )


def parse_and_align_slot_labels_list(
    token_ranges: List[Tuple[int, int]], slots_field: str, use_bio_labels: bool = False
):
    slots_field = slots_field or ""
    slot_list = parse_slot_string(slots_field)

    token_labels = []
    for t_start, t_end in token_ranges:
        tok_label = Slot.NO_LABEL_SLOT
        max_overlap = 0
        for s in slot_list:
            curr_overlap = s.token_overlap(t_start, t_end)
            if curr_overlap > max_overlap:
                max_overlap = curr_overlap
                tok_label = s.token_label(use_bio_labels, t_start, t_end)
        token_labels.append(tok_label)
    return token_labels


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
    if label.startswith(Slot.B_LABEL_PREFIX) or label.startswith(Slot.I_LABEL_PREFIX):
        label = label[len(Slot.B_LABEL_PREFIX) :]
    return label


def merge_token_labels_by_bio(token_ranges, labels):
    summary_list = []
    previous_B = None
    for i, label in enumerate(labels):
        # Take action only if the prefix is not i
        if not label.startswith(Slot.I_LABEL_PREFIX):
            # Label the previous chunk
            if previous_B is not None:
                begin = token_ranges[previous_B][0]
                end = token_ranges[i - 1][1]
                summary_list.append(
                    ":".join([str(begin), str(end), strip_bio_prefix(labels[i - 1])])
                )
            # Assign the begin location of new chunk
            if label.startswith(Slot.B_LABEL_PREFIX):
                previous_B = i
            else:  # label == Slot.NO_LABEL_SLOT
                previous_B = None

    # Take last token into account
    if previous_B is not None:
        begin = token_ranges[previous_B][0]
        end = token_ranges[-1][1]
        summary_list.append(
            ":".join([str(begin), str(end), strip_bio_prefix(labels[-1])])
        )

    return summary_list


def merge_token_labels_by_label(token_ranges, labels):
    # no bio prefix in labels
    begin = token_ranges[0][0]
    end = token_ranges[0][1]

    summary_list = []
    for i in range(1, len(labels)):
        # Extend
        if labels[i] == labels[i - 1] and labels[i] != Slot.NO_LABEL_SLOT:
            end = token_ranges[i][1]

        # Update and start new
        elif (
            (labels[i] != labels[i - 1])
            and (labels[i] != Slot.NO_LABEL_SLOT)
            and (labels[i - 1] != Slot.NO_LABEL_SLOT)
        ):
            summary_list.append(":".join([str(begin), str(end), labels[i - 1]]))
            begin = token_ranges[i][0]
            end = token_ranges[i][1]

        # Update and skip
        elif (
            (labels[i] != labels[i - 1])
            and (labels[i] == Slot.NO_LABEL_SLOT)
            and (labels[i - 1] != Slot.NO_LABEL_SLOT)
        ):
            summary_list.append(":".join([str(begin), str(end), labels[i - 1]]))

        # Skip
        elif (
            (labels[i] != labels[i - 1])
            and (labels[i] != Slot.NO_LABEL_SLOT)
            and (labels[i - 1] == Slot.NO_LABEL_SLOT)
        ):
            begin = token_ranges[i][0]
            end = token_ranges[i][1]

    # Take last token into account
    if labels[-1] != Slot.NO_LABEL_SLOT:
        summary_list.append(":".join([str(begin), str(end), labels[-1]]))

    return summary_list


def merge_token_labels_to_slot(token_ranges, labels, use_bio_label=True):
    # ensures that all labels, some of which may be SpecialToken tyeps,
    # are normalized to string for the metric reporter
    labels = [str(x) for x in labels]
    summary_list = (
        merge_token_labels_by_bio(token_ranges, labels)
        if use_bio_label
        else merge_token_labels_by_label(token_ranges, labels)
    )

    return ",".join(summary_list)


def is_number(string):
    try:
        float(string)
        return True
    except ValueError:
        pass

    try:
        unicodedata.numeric(string)
        return True
    except (TypeError, ValueError):
        pass

    return False


def unkify(token: str):
    res = "<unk>"
    for idx in range(len(token)):
        if token[idx].isdigit():
            res = "<unk>-NUM"

    return res

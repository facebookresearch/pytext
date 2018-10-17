#!/usr/bin/env python3
from typing import List, Any, Tuple
import json


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
            blabel_name = "{}{}".format(self.B_LABEL_PREFIX, self.label)
            ilabel_name = "{}{}".format(self.I_LABEL_PREFIX, self.label)
            if token_start == self.start and token_overlap:
                token_label = blabel_name
            elif token_start > self.start and token_overlap:
                token_label = ilabel_name
        else:
            if token_overlap:
                token_label = self.label
        return token_label

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


def parse_json_array(
    json_text: str
) -> List[str]:
    return json.loads(json_text)


# In order to process each field independently, we need to align slot labels
def align_slot_labels(
    tokenized_text: List[Tuple[str, Tuple[int, int]]],
    slots_field: str,
    use_bio_labels: bool = False,
):
    slot_list = parse_slot_string(slots_field)

    token_labels = []
    for (_, (t_start, t_end)) in tokenized_text:
        tok_label = Slot.NO_LABEL_SLOT
        max_overlap = 0
        for s in slot_list:
            curr_overlap = s.token_overlap(t_start, t_end)
            if curr_overlap > max_overlap:
                max_overlap = curr_overlap
                tok_label = s.token_label(use_bio_labels, t_start, t_end)
        token_labels.append(tok_label)
    return " ".join(token_labels)

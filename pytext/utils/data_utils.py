#!/usr/bin/env python3
import json
from typing import Any, Generator, List, NamedTuple, Tuple


def simple_tokenize(s: str) -> List[str]:
    return s.split(" ")


def no_tokenize(s: Any) -> Any:
    return s


class SlotBase(NamedTuple):
    label: str
    start: int
    end: int


class Slot(SlotBase):
    B_LABEL_PREFIX = "B-"
    I_LABEL_PREFIX = "I-"
    NO_LABEL_SLOT = "NoLabel"

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
        return f"{self.start}:{self.end}:{self.label}"


def parse_slot_string(label_str: str) -> Generator[Slot, None, None]:
    for token_label in label_str.split(","):
        label_elements = token_label.split(":", 2)
        if len(label_elements) < 3:
            # Invalid label, for now we swallow it and move on
            continue
        begin, end, label = label_elements
        yield Slot(label, int(begin), int(end))


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
    return " ".join(token_labels)

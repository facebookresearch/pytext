#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import json
import unicodedata
from typing import Any, List, Tuple

from pytext.common.constants import VocabMeta


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


def parse_json_array(json_text: str) -> List[str]:
    return json.loads(json_text)


# In order to process each field independently, we need to align slot labels
def align_slot_labels(
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
    return " ".join(token_labels)


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


# This function has been adapted from
# https://github.com/clab/rnng/blob/master/get_oracle.py
def unkify(token):  # noqa: C901
    if len(token.rstrip()) == 0:
        return VocabMeta.UNK_TOKEN

    numCaps = 0
    hasDigit = False
    hasDash = False
    hasLower = False
    for char in token.rstrip():
        if char.isdigit():
            hasDigit = True
        elif char == "-":
            hasDash = True
        elif char.isalpha():
            if char.islower():
                hasLower = True
            elif char.isupper():
                numCaps += 1
    result = VocabMeta.UNK_TOKEN
    lower = token.rstrip().lower()
    ch0 = token.rstrip()[0]
    if ch0.isupper():
        if numCaps == 1:
            result = result + "-INITC"
        else:
            result = result + "-CAPS"
    elif not (ch0.isalpha()) and numCaps > 0:
        result = result + "-CAPS"
    elif hasLower:
        result = result + "-LC"
    if hasDigit:
        result = result + "-NUM"
    if hasDash:
        result = result + "-DASH"
    if lower[-1] == "s" and len(lower) >= 3:
        ch2 = lower[-2]
        if not (ch2 == "s") and not (ch2 == "i") and not (ch2 == "u"):
            result = result + "-s"
    elif len(lower) >= 5 and not (hasDash) and not (hasDigit and numCaps > 0):
        if lower[-2:] == "ed":
            result = result + "-ed"
        elif lower[-3:] == "ing":
            result = result + "-ing"
        elif lower[-3:] == "ion":
            result = result + "-ion"
        elif lower[-2:] == "er":
            result = result + "-er"
        elif lower[-3:] == "est":
            result = result + "-est"
        elif lower[-2:] == "ly":
            result = result + "-ly"
        elif lower[-3:] == "ity":
            result = result + "-ity"
        elif lower[-1] == "y":
            result = result + "-y"
        elif lower[-2:] == "al":
            result = result + "-al"
    return result

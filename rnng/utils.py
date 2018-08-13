#!/usr/bin/env python3
from collections import defaultdict

UNKNOWN_WORD = "UNK"
INTENT_PREFIX = "IN:"
SLOT_PREFIX = "SL:"
COMBINATION_INTENT_LABEL = INTENT_PREFIX + "COMBINE"
COMBINATION_SLOT_LABEL = SLOT_PREFIX + "COMBINE"
SHIFT = "SHIFT"
REDUCE = "REDUCE"
NUM = "NUM"


class BiDict:
    def __init__(self):
        self.index_to_obj = defaultdict()
        self.obj_to_index = defaultdict()

    def check_add(self, ob):
        if ob not in self.obj_to_index:
            index = len(self.index_to_obj)
            self.index_to_obj[index] = ob
            self.obj_to_index[ob] = index
        return self.obj_to_index[ob]

    def check(self, ob) -> bool:
        return ob in self.obj_to_index

    def index(self, ob):
        return self.obj_to_index[ob]

    def value(self, ind):
        return self.index_to_obj[ind]

    def vocab(self):
        return self.obj_to_index.keys()

    def size(self):
        return len(self.index_to_obj.items())

    def get_sorted_objs(self, check_contiguous=True):

        if check_contiguous and sorted(self.index_to_obj) != [
            x for x in range(len(self.index_to_obj))
        ]:

            raise ValueError("BiDict indices not contiguous")

        return [v for (k, v) in sorted(self.index_to_obj.items())]


def is_valid_nonterminal(node_label: str) -> bool:
    return node_label.startswith(INTENT_PREFIX) or node_label.startswith(SLOT_PREFIX)


def is_intent_nonterminal(node_label: str) -> bool:
    return node_label.startswith(INTENT_PREFIX)


def is_slot_nonterminal(node_label: str) -> bool:
    return node_label.startswith(SLOT_PREFIX)


def is_unsupported(node_label: str) -> bool:
    return (
        is_intent_nonterminal(node_label)
        and node_label.lower().find("unsupported", 0) > 0
    )


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata

        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False

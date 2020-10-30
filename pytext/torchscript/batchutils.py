#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import List, Tuple


def max_tokens(per_sentence_tokens: List[List[Tuple[str, int, int]]]) -> int:
    """receive the tokenize output for a batch per_sentence_tokens,
    return the max token length of any sentence"""

    if len(per_sentence_tokens) == 0:
        return 0

    sentence_lengths = [len(sentence) for sentence in per_sentence_tokens]
    return max(sentence_lengths)

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List, Optional, Tuple


def extract_beam_subtrees(beam: List[List[str]]) -> List[List[str]]:
    return list(
        filter(
            lambda processed_pred: processed_pred is not None,
            map(lambda tree_pred: extract_subtree(tree_pred), beam),
        )
    )


def filter_invalid_beams(beam: List[List[str]]) -> List[List[str]]:
    return list(filter(lambda pred: is_valid_tree(pred), beam))


def is_valid_tree(beam: List[str]) -> bool:
    paren_stack: List[int] = []
    for i, token in enumerate(beam):
        if len(token) == 0:
            continue

        if token[0] == "[":
            paren_stack.append(i)
        elif token[-1] == "]":
            if len(paren_stack) == 0:
                return False
            paren_stack.pop()

    return len(paren_stack) == 0


def extract_subtree(beam: List[str]) -> Optional[List[str]]:
    paren_stack: List[int] = []
    longest_valid_tree: Tuple[int, int] = (-1, -1)

    # determine what the valid subtree is
    # in the prediction
    for i, token in enumerate(beam):
        if len(token) == 0:
            # skip over empty tokens since we check
            # characters in the token strings
            continue

        if token[0] == "[":
            # if the token begints with "[" it is the
            # start of a sequence so appending to stack
            paren_stack.append(i)
        elif token[-1] == "]":
            # found a potential end of a valid subsequence
            if len(paren_stack) == 0:
                # cannot find a valid tree
                # reset stack and continue
                paren_stack = []
                continue

            # the valid subtree is from current position to
            # the start sequence at `tree_start`
            tree_start: int = paren_stack.pop()
            if (i - tree_start) > (longest_valid_tree[1] - longest_valid_tree[0]):
                # if this subsequence is longer than the mosdt valid one
                # this is the longest valid subsequence
                longest_valid_tree = (tree_start, i)

    if longest_valid_tree == (-1, -1):
        # no valid sequence was found
        return None

    valid_tree_start, valid_tree_end = longest_valid_tree
    return beam[valid_tree_start : valid_tree_end + 1]

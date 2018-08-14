#!/usr/bin/env python3
from typing import Tuple, List, Set, Union

from pytext import metrics
from pytext.rnng.annotation import (
    Annotation,
    Intent,
    Node,
    Root,
    Slot,
    Token,
    Tree,
)
from pytext.shared_tokenizer import SharedTokenizer
from messenger.assistant.cu.core.ttypes import IntentFrame, FilledSlot, Span
from pytext.rnng.utils import INTENT_PREFIX, SLOT_PREFIX

tokenizer = SharedTokenizer()


def annotation_to_intent_frame(
    annotation: Annotation,
    utterance: str,
    token_spans: List[Tuple[str, Tuple[int, int]]],
    domain: str = "",
) -> IntentFrame:
    """
    Converts an RNNG Annotation into an IntentFrame. Useful for storing
    annotations in XDB or for serialization.

    Args:
        annotation:
            The RNNG Annotation tree.
        utterance:
            The original, pre-tokenized utterance.
        token_spans:
            A list of tokens that are used in the annotation, with the
            (start, end) spans of those tokens in the original utterance.
        domain:
            The domain of the outermost intent of the annotation.

    Returns:
        An IntentFrame object.
    """

    children = annotation.root.children
    if len(children) != 1:
        raise ValueError(
            "There should be one top-level ancestor node in the root annotation."
        )
    root = children[0]
    if not token_spans:
        raise ValueError("Token spans list was missing or empty, but is required.")

    return _node_to_intent_frame(root, utterance, token_spans, domain)


def _node_to_intent_frame(
    node: Node,
    utterance: str,
    token_spans: List[Tuple[str, Tuple[int, int]]],
    domain: str = "",
) -> IntentFrame:
    """
    Converts an Annotation node to an IntentFrame. To be used privately only.
    """
    if type(node) != Intent:
        raise ValueError(
            "The type of the node should be Intent, but is %s" % type(node)
        )

    start, end = _get_character_spans(node.list_tokens(), token_spans)
    return IntentFrame(
        domain=domain,
        utterance=utterance[start:end],
        intent=node.label,
        slots=[
            _node_to_filled_slot(node, utterance, token_spans, domain)
            for node in node.children
            if type(node) == Slot
        ],
        span=Span(start=start, end=end),
    )


def _node_to_filled_slot(
    node: Node,
    utterance: str,
    token_spans: List[Tuple[str, Tuple[int, int]]],
    domain: str = "",
) -> FilledSlot:
    """
    Converts an Annotation node to a FilledSlot. To be used privately only.
    """
    if type(node) != Slot:
        raise ValueError("The type of the node should be Slot, but is %s" % type(node))

    start, end = _get_character_spans(node.list_tokens(), token_spans)
    filled_slot = FilledSlot(
        text=utterance[start:end], id=node.label, span=Span(start=start, end=end)
    )

    children = [child for child in node.children if type(child) == Intent]
    if children:
        if len(children) != 1:
            raise ValueError("Slots can have at most one child Intent.")

        filled_slot.subframe = _node_to_intent_frame(
            children[0], filled_slot.text, _trim_token_spans(token_spans, filled_slot)
        )

    return filled_slot


def _trim_token_spans(
    token_spans: List[Tuple[str, Tuple[int, int]]], slot: FilledSlot
) -> List[Tuple[str, Tuple[int, int]]]:
    """
    Removes extraneous tokens and re-aligns spans. To be used privately only.
    """
    slot_start = slot.span.start
    slot_end = slot.span.end

    new_token_spans = []
    for (label, (start, end)) in token_spans:
        if start >= slot_start and end <= slot_end:
            new_token_spans.append((label, (start - slot_start, end - slot_start)))

    return new_token_spans


def _get_character_spans(
    tokens: List[Token], token_spans: List[Tuple[str, Tuple[int, int]]]
) -> Tuple[int, int]:
    """
    Get the character span of the list of tokens.
    tokens and token_spans are expected to be in sorted order.
    """
    ## Earlier dict implementation returned first occurrence of first and last token
    ## This returns first occurrence of (first:last) tokens,
    ## avoiding common cases like multiple 'the'
    ## TODO: Handle multiple occurrences of token list
    for i in range(len(token_spans)):
        if token_spans[i][0] == tokens[0]:
            if [item[0] for item in token_spans[i : i + len(tokens)]] == tokens:
                start = token_spans[i][1][0]
                end = token_spans[i + len(tokens) - 1][1][1]
                return start, end

    raise ValueError("Token substring not found.")


def intent_frame_to_tree(intent_frame: IntentFrame) -> Tree:
    """
    Converts an IntentFrame into an RNNG Tree

    Args:
        intent_frame:
            An IntentFrame object.
    Returns:
        The RNNG Tree.
    """

    root = Root()
    token_spans = tokenizer.tokenize_with_ranges(intent_frame.utterance)
    return Tree(
        _intent_frame_to_node(intent_frame, intent_frame.utterance, token_spans, root),
        True,
    )


def _intent_frame_to_node(
    intent_frame: IntentFrame,
    utterance: str,
    token_spans: List[Tuple[str, Tuple[int, int]]],
    parent: Node,
) -> Node:
    """
    Converts an IntentFrame node to a node. To be used privately only.
    """
    # T30990958 Update flat intents and slots to have IN: and SL: prefixes
    intent_node = Intent(
        ("" if intent_frame.intent.startswith(INTENT_PREFIX) else INTENT_PREFIX)
        + intent_frame.intent
    )
    intent_node.parent = parent

    slot_dict = {slot.span.start: slot for slot in intent_frame.slots}

    token_idx = 0
    while token_idx < len(token_spans):
        label, (start, end) = token_spans[token_idx]

        if start in slot_dict.keys():
            filled_slot = slot_dict[start]
            # T30990958 Update flat intents and slots to have IN: and SL: prefixes
            slot_node = Slot(
                ("" if filled_slot.id.startswith(SLOT_PREFIX) else SLOT_PREFIX)
                + filled_slot.id
            )
            slot_node.parent = intent_node

            if filled_slot.subframe:
                subframe_start = (
                    filled_slot.span.start + filled_slot.subframe.span.start
                )
                subframe_end = filled_slot.span.start + filled_slot.subframe.span.end

                slot_node = _intent_frame_to_node(
                    filled_slot.subframe,
                    utterance[subframe_start:subframe_end],
                    _trim_token_spans(token_spans, filled_slot),
                    slot_node,
                )
                while end < subframe_end:
                    token_idx = token_idx + 1
                    label, (start, end) = token_spans[token_idx]

            else:
                slot_node = _insert_token(slot_node, label)

            while end < filled_slot.span.end:
                token_idx = token_idx + 1
                label, (start, end) = token_spans[token_idx]
                slot_node = _insert_token(slot_node, label)

            intent_node.children.append(slot_node)

        else:
            intent_node = _insert_token(intent_node, label)

        token_idx = token_idx + 1

    parent.children.append(intent_node)
    return parent


def _insert_token(parent: Node, label: str) -> Node:
    token_node = Token(label, 0)  # token_count is irrelevant
    token_node.parent = parent
    parent.children.append(token_node)
    return parent


def tree_to_metric_node(tree: Tree) -> metrics.Node:
    """
    Creates a generic node (i.e., metrics.Node) from tree assuming the utterance
    is a concatenation of the tokens by whitespaces. This is used in RNNG model
    evalution to convert trees into frames to be consumed by metric computation.
    The function does not necessarily reproduce the indices in the original utterance
    as extra whitespaces can be introduced, so its use should be strictly limited to
    the above mentioned case.
    """
    return _node_to_metrics_node(tree.root.children[0])


def _node_to_metrics_node(node: Union[Intent, Slot], start: int = 0) -> metrics.Node:
    """
    input: start is the absolute start position in utterance
    """
    res_children: Set[metrics.Node] = set()
    idx = start
    for child in node.children:
        if type(child) == Token:
            idx += len(child.label) + 1
        elif type(child) == Intent or type(child) == Slot:
            res_child = _node_to_metrics_node(child, idx)
            res_children.add(res_child)
            idx = res_child.span.end + 1
        else:
            raise ValueError("Child must be Token, Intent or Slot!")
    node = metrics.Node(
        label=node.label, span=metrics.Span(start, idx - 1), children=res_children
    )
    return node


# TODO: (wenfangxu) T32687283 remove this function when we have more elegant way
#       of converting tree to intent frame in the compositional workflow.
def tree_to_intent_frame(tree: Tree) -> IntentFrame:
    """
    Creates intent frame from tree by concatenating the tokens in the tree with
    whitespaces. This is used in RNNG model evalution to convert trees into intent
    frames to be consumed by metric computation. The function does not necessarily
    reproduce the original utterance as extra whitespaces can be introduced, so its
    use should be strictly limited to the above mentioned case.
    """
    return _intent_to_intent_frame(tree.root.children[0])


def _intent_to_intent_frame(intent: Intent, start: int = 0) -> IntentFrame:
    """
    input: start is the relative start position in parent FilledSlot
    """
    intent_frame = IntentFrame(intent=intent.label, slots=[])
    tokens: List[str] = []
    idx = 0
    for child in intent.children:
        if type(child) == Token:
            tokens.append(child.label)
            idx += len(child.label) + 1
        elif type(child) == Slot:
            filled_slot = _slot_to_filled_slot(child, idx)
            intent_frame.slots.append(filled_slot)
            idx = filled_slot.span.end + 1
            tokens.append(filled_slot.text)
        else:
            raise ValueError("Intent has child other than Slot or Token!")
    intent_frame.span = Span(start, start + idx - 1)
    intent_frame.utterance = " ".join(tokens)
    return intent_frame


def _slot_to_filled_slot(slot: Slot, start: int) -> FilledSlot:
    """
    input: start is the relative start position in parent IntentFrame
    """
    filled_slot = FilledSlot(id=slot.label)
    tokens: List[str] = []
    idx = 0
    for child in slot.children:
        if type(child) == Token:
            tokens.append(child.label)
            idx += len(child.label) + 1
        elif type(child) == Intent:
            intent_frame = _intent_to_intent_frame(child, idx)
            filled_slot.subframe = intent_frame
            idx = intent_frame.span.end + 1
            tokens.append(intent_frame.utterance)
        else:
            raise ValueError("Slot has child other than Intent or Token!")
    filled_slot.text = " ".join(tokens)
    if idx - 1 != len(filled_slot.text):
        raise Exception("Index and length of concatenated tokens do not match!")
    filled_slot.span = Span(start=start, end=start + idx - 1)
    return filled_slot

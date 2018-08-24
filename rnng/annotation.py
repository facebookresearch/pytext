#!/usr/bin/env python3
from typing import Any, List, Tuple

from assistant.lib.feat.ttypes import ModelFeatures
from pytext.data.shared_featurizer import SharedFeaturizer
from pytext.rnng.utils import (
    COMBINATION_INTENT_LABEL,
    COMBINATION_SLOT_LABEL,
    INTENT_PREFIX,
    REDUCE,
    SHIFT,
    SLOT_PREFIX,
    BiDict,
    is_intent_nonterminal,
    is_slot_nonterminal,
    is_valid_nonterminal,
)


"""
A data structure for an Intents and Slots annotation:
Each Node has type (Root, Intent, or Token), a pointer to its parent, and
a list of its children. Token's children == None

Annotation.validate_tree() will check for valid nesting.

However, this class is intended not for validation but for annotator agreement
calculations.
"""


class Annotation:
    def __init__(
        self,
        annotation_string: str,
        brackets: str = "[]",
        combination_labels: bool = True,
        add_dict_feat: bool = False,
        accept_flat_intents_slots: bool = False,
    ) -> None:
        super(Annotation, self).__init__()
        self.OPEN = brackets[0]
        self.CLOSE = brackets[1]
        self.combination_labels = combination_labels
        self.model_feats = None

        # Expected annotation_string (tab-separated):
        # intent, slots, utterance, sparse_feat, seqlogical
        # OR only seqlogical
        parts = annotation_string.rstrip().split("\t")
        if len(parts) == 5:
            [_, _, utterance, sparse_feat, self.seqlogical] = parts
            self.utterance = utterance
            if add_dict_feat:
                self.model_feats: ModelFeatures = SharedFeaturizer().featurize(
                    utterance, sparse_feat
                )
                assert self.model_feats is not None
        elif len(parts) == 1:
            [self.seqlogical] = parts
        else:
            raise ValueError("Cannot parse annotation_string")

        self.items = self.split_seqlogical()
        self.tree = Tree(
            self.build_tree(accept_flat_intents_slots),
            combination_labels
        )
        self.root: Root = self.tree.root

    def split_seqlogical(self):
        if not self.seqlogical.endswith("]"):
            raise ValueError("Not a valid tree.")

        result = []
        previous_char = ""
        for char in self.seqlogical:
            if char == self.OPEN and previous_char != "\\":
                result.append(" ")
                result.append(char)
            elif char == self.CLOSE and previous_char != "\\":
                result.append(" ")
                result.append(char)
                result.append(" ")
            else:
                result.append(char)
            previous_char = char

        return "".join(result).split()

    # add parameter to turn off the added build method
    def build_tree(self, accept_flat_intents_slots: bool = False):  # noqa
        token_count = 0
        root = Root()

        node_stack: List[Any] = [root]
        num_topintent = 0

        for item in self.items:
            if item == self.CLOSE:
                # close the non-terminal
                if not node_stack:
                    raise ValueError("Not a valid tree.")
                node_stack.pop()
            elif item.startswith(self.OPEN):
                if len(node_stack) < 1:
                    raise ValueError("Not a valid tree.")

                # either intent or slot non-terminal
                label = item[1:]
                if item.startswith(self.OPEN + INTENT_PREFIX):
                    node_stack.append(Intent(label))
                elif item.startswith(self.OPEN + SLOT_PREFIX):
                    node_stack.append(Slot(label))
                else:
                    if accept_flat_intents_slots:
                        # Temporary, for compatibility with flat annotations
                        # that does not contain IN:, SL: prefixes
                        #
                        # This assumes any child of ROOT or SLOT must be INTENT,
                        # and any child of INTENT must be SLOT
                        if type(node_stack[-1]) == Root or \
                                type(node_stack[-1]) == Slot:
                            node_stack.append(Intent(label))
                        elif type(node_stack[-1]) == Intent:
                            node_stack.append(Slot(label))
                        else:
                            raise ValueError(
                                'The previous object in node_stack is not of'
                                + ' type Root, Intent or Slot.'
                            )
                    else:
                        raise ValueError(
                            "Label "
                            + item
                            + " must start with IN: or SL: for "
                            + str(self.seqlogical)
                        )

                if len(node_stack) < 2:
                    raise ValueError("Not a valid tree.")

                node_stack[-1].parent = node_stack[-2]
                node_stack[-2].children.append(node_stack[-1])

                if type(node_stack[-1]) == Intent and node_stack[-1].parent == root:
                    num_topintent += 1

            else:
                # Token terminal
                token = Token(item, token_count)
                # self.tokenNode_list.append(token)
                token_count += 1
                if len(node_stack) <= 0:
                    raise ValueError("Not a valid tree.")
                token.parent = node_stack[-1]
                node_stack[-1].children.append(token)

        if len(node_stack) > 1:
            raise ValueError("Not a valid tree.")

        if num_topintent > 1 and self.combination_labels:
            comb_intent = Intent(COMBINATION_INTENT_LABEL)
            node_stack.insert(1, comb_intent)
            for c in root.children:
                if type(c) == Intent:
                    comb_slot = Slot(COMBINATION_SLOT_LABEL)
                    comb_slot.parent = comb_intent
                    comb_slot.children.append(c)
                    comb_intent.children.append(comb_slot)
                    c.parent = comb_slot
                else:
                    c.parent = comb_intent
                    comb_intent.children.append(c)
            comb_intent.parent = root
            root.children = [comb_intent]

        return root

    def __str__(self):
        """
        A tab-indented version of the tree.
        strip() removes an extra final newline added during recursion
        """
        return self.tree.__str__()

    def __eq__(self, other):
        return self.tree == other.tree


class Node:
    def __init__(self, label):
        self.label = label  # The name of the intent, slot, or token
        self.children = []  # the children of this node (Intent, Slot, or Token)
        self.parent = None

    def list_ancestors(self):
        ancestors = []
        if self.parent:
            if type(self.parent) != Root:
                ancestors.append(self.parent)
                ancestors += self.parent.list_ancestors()
        return ancestors

    def validate_node(self):
        if self.children:
            for child in self.children:
                child.validate_node()

    # Returns all tokens in the span covered by this node
    def list_tokens(self):
        tokens = []
        if self.children:
            for child in self.children:
                if type(child) == Token:
                    tokens.append(child.label)
                else:
                    tokens += child.list_tokens()
        return tokens

    def get_token_span(self):
        """
        0 indexed
        Like array slicing: For the first 3 tokens, returns 0, 3
        """
        indices = self._get_token_indices()
        if len(indices) > 0:
            return min(indices), max(indices) + 1
        else:
            return None

    def _get_token_indices(self):
        indices = []
        if self.children:
            for child in self.children:
                if type(child) == Token:
                    indices.append(child.index)
                else:
                    indices += child._get_token_indices()
        return indices

    def list_nonTerminals(self):
        """
        Returns all Intent and Slot nodes subordinate to this node
        """
        non_terminals = []
        for child in self.children:
            if type(child) != Root and type(child) != Token:
                non_terminals.append(child)
                non_terminals += child.list_nonTerminals()
        return non_terminals

    def list_terminals(self):
        """
        Returns all Token nodes
        """
        terminals = []
        for child in self.children:
            if type(child) == Token:
                terminals.append(child)
            else:
                terminals += child.list_terminals()
        return terminals

    def get_info(self):
        if type(self) == Token:
            return Token_Info(self)
        return Node_Info(self)

    def flat_str(self):
        string = ""
        if type(self) == Intent or type(self) == Slot:
            string = "["
        if type(self) != Root:
            string += str(self.label) + " "
        if self.children:
            for child in self.children:
                string += child.flat_str()
        if type(self) == Intent or type(self) == Slot:
            string += "] "
        return string

    def children_flat_str_spans(self):
        string = str(self.get_token_span()) + ":"
        if self.children:
            for child in self.children:
                string += child.flat_str()
        return string

    def __str__(self):
        string = self._recursive_str("", "")
        return string

    def _recursive_str(self, string, spacer):
        string = spacer + str(self.label) + "\n"
        spacer += "\t"

        if self.children:
            for child in self.children:
                string += child._recursive_str(string, spacer)
        return string

    def __eq__(self, other):
        return self.label == other.label and self.children == other.children


class Root(Node):
    def __init__(self):
        super().__init__("ROOT")

    def validate_node(self):
        super().validate_node()
        for child in self.children:
            if type(child) == Slot or type(child) == Root:
                raise TypeError(
                    "A root child must be an intent or token: " + self.label
                )
            elif self.parent is not None:
                raise TypeError("A root should not have a parent: " + self.label)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.children == other.children


class Intent(Node):
    def __init__(self, label):
        super().__init__(label)

    def validate_node(self):
        super().validate_node()
        for child in self.children:
            if type(child) == Intent or type(child) == Root:
                raise TypeError(
                    "An intent child must be a slot or token: " + self.label
                )

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.label == other.label and self.children == other.children


class Slot(Node):
    def __init__(self, label):
        super().__init__(label)

    def validate_node(self):
        super().validate_node()
        for child in self.children:
            if type(child) == Slot or type(child) == Root:
                raise TypeError(
                    "An slot child must be an intent or token: " + self.label
                )

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.label == other.label and self.children == other.children


class Token(Node):
    def __init__(self, label, index):
        super().__init__(label)
        self.index = index
        self.children = None

    def validate_node(self):
        if self.children is not None:
            raise TypeError(
                "A token node is terminal and should not \
                    have children: "
                + self.label
                + " "
                + str(self.children)
            )

    def remove(self):
        """
        Removes this token from the tree
        """
        self.parent.children.remove(self)
        self.parent = None

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.label == other.label and self.index == other.index


class Token_Info:
    """
    This class extracts the essential information for a token for use in rules.
    """

    def __init__(self, node):

        self.token_word = node.label

        self.parent_label = self.get_parent(node)

        self.ancestors = [a.label for a in node.list_ancestors()]

        self.prior_token = None
        prior = node.get_prior_token()
        if prior:
            self.prior_token = prior.label

        self.next_token = None
        next_token = node.get_next_token()
        if next_token:
            self.next_token = next_token.label

    def get_parent(self, node):
        if node.parent and type(node.parent) != Root:
            return node.parent.label
        return None

    def __str__(self):
        result = []
        result.append("Token Info:")
        result.append("Token Word: " + self.token_word)
        result.append("Previous Token: " + str(self.prior_token))
        result.append("Next Token: " + str(self.next_token))
        result.append("Parent: " + str(self.parent_label))
        result.append("Ancestors: " + ", ".join(self.ancestors))
        return "\n".join(result)


class Node_Info:
    """
    This class extracts the essential information for a mode, for use in rules.
    """

    def __init__(self, node):
        self.label = node.label
        # This is all descendent tokens, not just immediate children tokens
        self.tokens = node.list_tokens()
        # If no parent, None
        self.parent_label = self.get_parent(node)

        # only look at slot or intent children
        self.children = []
        for a in node.children:
            if (type(a) == Slot) or (type(a) == Intent):
                self.children.append(a.label)

        self.token_indices = node._get_token_indices()
        self.ancestors = [a.label for a in node.list_ancestors()]
        # This is only non-temrinal descendents. Did you want tokens too?
        self.descendents = [d.label for d in node.list_nonTerminals()]
        self.prior_token = None
        prior = node.get_prior_token()
        if prior:
            self.prior_token = prior.label
        if type(node) == Token:
            self.label_type = "TOKEN"
        elif type(node) == Intent:
            self.label_type = "INTENT"
        elif type(node) == Slot:
            self.label_type = "SLOT"

        # same span as parent
        self.same_span = self.get_same_span(node)

    def get_same_span(self, node):
        if node.parent:
            if set(node.parent.list_tokens()) == set(node.list_tokens()):
                return True
        return False

    def get_parent(self, node):
        if node.parent and type(node.parent) != Root:
            return node.parent.label
        return None

    def __str__(self):
        result = []
        result.append("Info:")
        result.append("Label: " + self.label)
        result.append("Tokens: " + " ".join(self.tokens))
        result.append(
            "Token Indicies: " + ", ".join([str(i) for i in self.token_indices])
        )
        result.append("Prior Token: " + str(self.prior_token))
        result.append("Parent: " + str(self.parent_label))
        result.append("Children: " + ", ".join(self.children))
        result.append("Ancestors: " + ", ".join(self.ancestors))
        result.append("Descendents: " + ", ".join(self.descendents))
        result.append("Label Type: " + str(self.label_type))
        result.append("Same Span: " + str(self.same_span))
        return "\n".join(result)


class Tree:
    def __init__(self, root: Root, combination_labels: bool) -> None:
        self.root = root
        self.combination_labels = combination_labels
        try:
            self.validate_tree()
        except ValueError as v:
            raise ValueError("Tree validation failed: {}".format(v))

    def validate_tree(self):
        """
        This is a method for checking that roots/intents/slots are
        nested correctly.
        Root( Intent( Slot( Intent( Slot, etc.) ) ) )
        """

        try:
            if self.combination_labels and not len(self.root.children) == 1:
                raise ValueError(
                    """Root should always have one child and not {}.
                    Look into {} and {}""".format(
                        len(self.root.children),
                        COMBINATION_INTENT_LABEL,
                        COMBINATION_SLOT_LABEL,
                    )
                )
            self.recursive_validation(self.root)
        except TypeError as t:
            raise ValueError(
                "Failed validation for {}".format(self.root) + "\n" + str(t)
            )

    def recursive_validation(self, node):
        node.validate_node()
        for child in node.children:
            child.validate_node()

    def print_tree(self):
        print(self.flat_str())

    def flat_str(self):
        return self.root.flat_str()

    def list_tokens(self):
        return self.root.list_tokens()

    def depth(self):
        # note that this calculation includes Root as part of the tree
        return self._depth(self.root)

    def _depth(self, n):
        if n.children:
            depths = []
            for c in n.children:
                depths.append(self._depth(c))
            return max(depths) + 1
        else:
            return 0

    def to_actions(self):
        actions = []
        self._to_actions(self.root, actions)
        return actions

    def _to_actions(self, node: Node, actions: List[str]):
        if type(node) == Token:
            actions.append(SHIFT)
            return

        for child in node.children:
            if not type(child) is Token:
                actions.append(child.label)
                self._to_actions(child, actions)
                actions.append(REDUCE)
            else:
                self._to_actions(child, actions)

    def __str__(self):
        """
        A tab-indented version of the tree.
        strip() removes an extra final newline added during recursion
        """
        return self.root.__str__().strip()

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.root == other.root


class TreeBuilder:
    def __init__(self, combination_labels: bool = True) -> None:
        self.combination_labels = combination_labels
        self.root = Root()
        self.node_stack = [self.root]
        self.token_count = 0
        self.finalzed = False

    def update_tree(self, action, label):
        assert not self.finalzed, "Cannot update tree since it's finalized"
        if action == REDUCE:
            self.node_stack.pop()
        elif is_valid_nonterminal(action):
            if is_intent_nonterminal(action):
                self.node_stack.append(Intent(label))
            elif is_slot_nonterminal(action):
                self.node_stack.append(Slot(label))
            else:
                raise ValueError("Don't understand action %s" % (action))

            self.node_stack[-1].parent = self.node_stack[-2]
            self.node_stack[-2].children.append(self.node_stack[-1])
        elif action == SHIFT:
            token = Token(label, self.token_count)
            self.token_count += 1
            token.parent = self.node_stack[-1]
            self.node_stack[-1].children.append(token)
        else:
            raise ValueError("Don't understand action %s" % (action))

    def finalize_tree(self):
        return Tree(self.root, self.combination_labels)


def tree_from_actions(tokens_str: List[str], actions_dict: BiDict, actions_indices):
    builder = TreeBuilder()
    i = 0
    for action_idx in actions_indices:
        action = actions_dict.value(action_idx)
        if action == REDUCE:
            builder.update_tree(action, None)
        elif action == SHIFT:
            builder.update_tree(action, tokens_str[i])
            i += 1
        else:
            builder.update_tree(action, action)
    tree = builder.finalize_tree()
    return tree


def list_from_actions(tokens_str: List[str], actions_dict: BiDict, actions_indices):
    actions_list: List[Tuple[str, Any]] = []
    i = 0
    for action_idx in actions_indices:
        action = actions_dict.value(action_idx)
        if action == SHIFT:
            actions_list.append((action, tokens_str[i]))
            i += 1
        elif is_intent_nonterminal(action):
            actions_list.append((INTENT_PREFIX, action.split(INTENT_PREFIX)[1]))
        elif is_slot_nonterminal(action):
            actions_list.append((SLOT_PREFIX, action.split(SLOT_PREFIX)[1]))
        else:
            actions_list.append((action, ""))
    return actions_list


def tree_from_tokens_and_indx_actions(
    tokens_str, actions_dict: BiDict, actions_indices
):
    builder = TreeBuilder()
    i = 0
    for action_idx in actions_indices:
        action = actions_dict.value(action_idx)
        if action == REDUCE:
            builder.update_tree(action, None)
        elif action == SHIFT:
            builder.update_tree(action, tokens_str[i])
            i += 1
        else:
            builder.update_tree(action, action)
    tree = builder.finalize_tree()
    return tree


def tree_from_indx_actions(
    token_dict: BiDict, token_indices, actions_dict: BiDict, actions_indices
):
    tokens_str = [token_dict.value(token_i) for token_i in token_indices]
    return tree_from_tokens_and_indx_actions(tokens_str, actions_dict, actions_indices)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        for i in range(1, len(sys.argv)):
            annot = Annotation(sys.argv[i])
            for child in annot.root.children:
                if type(child) == Intent:
                    print(child)

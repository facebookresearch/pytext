#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Any, List, Sized, Tuple

import torch as torch
import torch.nn as nn
from pytext.utils.cuda_utils import xaviervar


# token/non-terminal/sub-tree element on a stack.
# need this value for computing valid actions
class Element:
    def __init__(self, node) -> None:
        self.node = node

    def __str__(self):
        return str(self.node)

    def __repr__(self):
        return self.__str__()


class StackLSTM(Sized):
    def __init__(self, rnn, initial_state, p_empty_embedding):
        self.rnn = rnn
        self.list = (
            [(initial_state, (self._rnn_get_output(initial_state), "Root"))]
            if initial_state
            else None
        )
        self.empty = p_empty_embedding

    def _rnn_get_output(self, state):
        return state[0][-1]

    def push(self, expr, ele: Element) -> None:
        # assuming expr is always one element at a time; not a sequence.
        # making it a sequence since LSTM takes a sequence
        expr = expr.unsqueeze(1)
        output, new_embedding = self.rnn(expr, self.list[-1][0])
        self.list.append((new_embedding, (self._rnn_get_output(new_embedding), ele)))

    def pop(self) -> Tuple[Any, Element]:
        # returning tuple of out embedding and the name of the element
        return self.list.pop()[1]

    def top(self) -> Tuple[Any, Element]:
        return self.list[-1][1]

    def embedding(self):
        return (
            self._rnn_get_output(self.list[-1][0]) if len(self.list) > 1 else self.empty
        )

    def first_ele_match(self, funct):
        for st in self.list[::-1]:
            if funct(st):
                return st[1][1]
        return None

    def ele_from_top(self, index: int) -> Element:
        return self.list[len(self.list) - index - 1][1][1]

    def __len__(self):
        return len(self.list) - 1

    def __str__(self):
        return "->".join([str(x[1][1]) for x in self.list])

    def copy(self):
        other = StackLSTM(self.rnn, None, self.empty)
        other.list = list(self.list)
        return other


class CompositionFunction(nn.Module):
    def __init__(self):
        super().__init__()


class CompositionalNN(CompositionFunction):
    def __init__(self, lstm_dim):
        super().__init__()
        self.lstm_dim = lstm_dim
        self.lstm_fwd = nn.LSTM(lstm_dim, lstm_dim, 1)
        self.lstm_rev = nn.LSTM(lstm_dim, lstm_dim, 1)
        self.linear_seq = nn.Sequential(nn.Linear(2 * lstm_dim, lstm_dim), nn.Tanh())

    def forward(self, x):
        """
        Embed the sequence. If the input corresponds to [IN:GL where am I at]:
        - x will contain the embeddings of [at I am where IN:GL] in that order.
        - Forward LSTM will embed the sequence [IN:GL where am I at].
        - Backward LSTM will embed the sequence [IN:GL at I am where].
        The final hidden states are concatenated and then projected.

        Args:
            x: Embeddings of the input tokens in *reversed* order
        """
        # reset hidden every time
        lstm_hidden_fwd = (
            xaviervar(1, 1, self.lstm_dim),
            xaviervar(1, 1, self.lstm_dim),
        )
        lstm_hidden_rev = (
            xaviervar(1, 1, self.lstm_dim),
            xaviervar(1, 1, self.lstm_dim),
        )
        nt_element = x[-1]
        rev_rest = x[:-1]
        # Always put nt_element at the front
        fwd_input = [nt_element] + rev_rest[::-1]
        rev_input = [nt_element] + rev_rest
        stacked_fwd = self.lstm_fwd(torch.stack(fwd_input), lstm_hidden_fwd)[0][0]
        stacked_rev = self.lstm_rev(torch.stack(rev_input), lstm_hidden_rev)[0][0]
        combined = torch.cat([stacked_fwd, stacked_rev], dim=1)
        subtree_embedding = self.linear_seq(combined)
        return subtree_embedding


class CompositionalSummationNN(CompositionFunction):
    def __init__(self, lstm_dim):
        super().__init__()
        self.lstm_dim = lstm_dim

        self.linear_seq = nn.Sequential(nn.Linear(lstm_dim, lstm_dim), nn.Tanh())

    def forward(self, x):
        combined = torch.sum(torch.cat(x, dim=0), dim=0, keepdim=True)
        subtree_embedding = self.linear_seq(combined)
        return subtree_embedding


class ParserState:
    # Copies another state instead if supplied
    def __init__(self, parser=None):

        if not parser:
            return
        # Otherwise initialize normally
        self.buffer_stackrnn = StackLSTM(
            parser.buff_rnn, parser.init_lstm(), parser.pempty_buffer_emb
        )
        self.stack_stackrnn = StackLSTM(
            parser.stack_rnn, parser.init_lstm(), parser.empty_stack_emb
        )
        self.action_stackrnn = StackLSTM(
            parser.action_rnn, parser.init_lstm(), parser.empty_action_emb
        )

        self.predicted_actions_idx = []
        self.action_scores = []

        self.num_open_NT = 0
        self.is_open_NT: List[bool] = []
        self.found_unsupported = False

        # negative cumulative log prob so sort(states) is in descending order
        self.neg_prob = 0

    def finished(self):
        return len(self.stack_stackrnn) == 1 and len(self.buffer_stackrnn) == 0

    def copy(self):
        other = ParserState()
        other.buffer_stackrnn = self.buffer_stackrnn.copy()
        other.stack_stackrnn = self.stack_stackrnn.copy()
        other.action_stackrnn = self.action_stackrnn.copy()
        other.predicted_actions_idx = self.predicted_actions_idx.copy()
        other.action_scores = self.action_scores.copy()
        other.num_open_NT = self.num_open_NT
        other.is_open_NT = self.is_open_NT.copy()
        other.neg_prob = self.neg_prob
        other.found_unsupported = self.found_unsupported
        return other

    def __gt__(self, other):
        return self.neg_prob > other.neg_prob

    def __eq__(self, other):
        return self.neg_prob == other.neg_prob

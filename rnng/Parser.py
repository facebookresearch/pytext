#!/usr/bin/env python3

from typing import Sized, List, Tuple, Any

import numpy as np
import torch as torch
import torch.nn as nn
import torch.nn.functional as F

from pytext.rnng.config import CompositionalType
from pytext.config import PyTextConfig
from pytext.utils.cuda_utils import Variable, xaviervar
from pytext.rnng.utils import (
    BiDict,
    SHIFT,
    REDUCE,
    is_valid_nonterminal,
    is_slot_nonterminal,
    is_intent_nonterminal,
    is_unsupported,
)
from pytext.models.configs.embedding_config import DictEmbeddingConfig
from pytext.models.embeddings.dict_embedding import DictEmbedding
from pytext.rnng.ontology_constraint import OntologyConstraint

EMPTY_BIDICT = BiDict()


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


def get_one_hot_vector(tokens_list, num_tokens_vocab):
    one_hot = np.zeros((len(tokens_list), num_tokens_vocab))
    one_hot[np.arange(len(tokens_list)), tokens_list] = 1

    return torch.FloatTensor(one_hot).view(len(tokens_list), -1)


class CompositionalNN(nn.Module):
    def __init__(self, lstm_dim):
        super(CompositionalNN, self).__init__()
        self.lstm_dim = lstm_dim
        self.lstm_fwd = nn.LSTM(lstm_dim, lstm_dim, 1)
        self.lstm_rev = nn.LSTM(lstm_dim, lstm_dim, 1)
        self.linear_seq = nn.Sequential(nn.Linear(2 * lstm_dim, lstm_dim), nn.Tanh())

    def forward(self, x):
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
        rest = x[:-1]
        stacked_fwd = self.lstm_fwd(torch.stack(x), lstm_hidden_fwd)[0][0]
        stacked_rev = self.lstm_rev(torch.stack(rest + [nt_element]), lstm_hidden_rev)[
            0
        ][0]
        combined = torch.cat([stacked_fwd, stacked_rev], dim=1)
        subtree_embedding = self.linear_seq(combined)
        return subtree_embedding


class CompositionalSummationNN(nn.Module):
    def __init__(self, lstm_dim):
        super(CompositionalSummationNN, self).__init__()
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


class RNNGParser(nn.Module):
    def __init__(
        self,
        config: PyTextConfig,
        terminal_bidict: BiDict,
        actions_bidict: BiDict,
        dictfeat_bidict: BiDict = EMPTY_BIDICT,
    ) -> None:
        super(RNNGParser, self).__init__()

        self.pytext_config = config
        self.terminal_bidict = terminal_bidict
        self.actions_bidict = actions_bidict

        rnng_config = config.jobspec.model
        self.config = rnng_config
        self.constraints = rnng_config.constraints
        if self.constraints.ontology:
            self.ontology_constraint = OntologyConstraint(
                self.constraints.ontology, self.actions_bidict,
            )

        self.shift_idx: int = actions_bidict.index(SHIFT)
        self.reduce_idx: int = actions_bidict.index(REDUCE)

        # unsupported instances
        self.ignore_subNTs_roots: List[int] = [
            actions_bidict.index(nt)
            for nt in actions_bidict.vocab()
            if is_unsupported(nt)
        ]
        self.valid_NT_idxs: List[int] = [
            actions_bidict.index(nt)
            for nt in actions_bidict.vocab()
            if is_valid_nonterminal(nt)
        ]
        self.valid_IN_idxs: List[int] = [
            actions_bidict.index(nt)
            for nt in actions_bidict.vocab()
            if is_intent_nonterminal(nt)
        ]
        self.valid_SL_idxs: List[int] = [
            actions_bidict.index(nt)
            for nt in actions_bidict.vocab()
            if is_slot_nonterminal(nt)
        ]

        if rnng_config.compositional_type == CompositionalType.SUM:
            self.p_compositional = CompositionalSummationNN(
                lstm_dim=rnng_config.lstm.lstm_dim
            )
        elif rnng_config.compositional_type == CompositionalType.BLSTM:
            self.p_compositional = CompositionalNN(lstm_dim=rnng_config.lstm.lstm_dim)
        else:
            raise ValueError(
                "Cannot understand compositional flag {}".format(
                    rnng_config.compositional_type
                )
            )

        lstm_count = (
            rnng_config.ablation.use_buffer
            + rnng_config.ablation.use_stack
            + rnng_config.ablation.use_action
        )
        if lstm_count == 0:
            raise ValueError("Need atleast one of the Parser ablation flags to be True")

        self.action_linear = nn.Sequential(
            nn.Linear(
                lstm_count * rnng_config.lstm.lstm_dim, rnng_config.lstm.lstm_dim
            ),
            nn.ReLU(),
            nn.Linear(rnng_config.lstm.lstm_dim, self.actions_bidict.size()),
        )
        self.dropout_layer = nn.Dropout(p=rnng_config.dropout)
        embed_dim = config.jobspec.features.word_feat.embed_dim
        if config.jobspec.features.dict_feat:
            embed_dim += config.jobspec.features.dict_feat.embed_dim
        self.buff_rnn = nn.LSTM(
            embed_dim,
            rnng_config.lstm.lstm_dim,
            num_layers=rnng_config.lstm.num_layers,
            dropout=rnng_config.dropout,
        )
        self.stack_rnn = nn.LSTM(
            rnng_config.lstm.lstm_dim,
            rnng_config.lstm.lstm_dim,
            num_layers=rnng_config.lstm.num_layers,
            dropout=rnng_config.dropout,
        )
        self.action_rnn = nn.LSTM(
            rnng_config.lstm.lstm_dim,
            rnng_config.lstm.lstm_dim,
            num_layers=rnng_config.lstm.num_layers,
            dropout=rnng_config.dropout,
        )

        self.pempty_buffer_emb = nn.Parameter(torch.randn(1, rnng_config.lstm.lstm_dim))
        self.empty_stack_emb = nn.Parameter(torch.randn(1, rnng_config.lstm.lstm_dim))
        self.empty_action_emb = nn.Parameter(torch.randn(1, rnng_config.lstm.lstm_dim))

        self.WORDS_LOOKUP = nn.Embedding(
            terminal_bidict.size(), config.jobspec.features.word_feat.embed_dim
        )

        self.add_dict_feat = (
            dictfeat_bidict.size() > 0 and config.jobspec.features.dict_feat
        )
        if self.add_dict_feat:
            self.DICT_FEAT_LOOKUP = DictEmbedding(
                self._get_dict_feat_config(config, dictfeat_bidict.size())
            )

        self.criterion = nn.CrossEntropyLoss()

        self.ACTIONS_LOOKUP = nn.Embedding(
            self.actions_bidict.size(), rnng_config.lstm.lstm_dim
        )

    def init_word_weights(self, pretrained_word_weights):
        if pretrained_word_weights is not None:
            self.WORDS_LOOKUP.weight.data.copy_(pretrained_word_weights)

    # Initializes LSTM parameters
    def init_lstm(self):
        return (
            xaviervar(self.config.lstm.num_layers, 1, self.config.lstm.lstm_dim),
            xaviervar(self.config.lstm.num_layers, 1, self.config.lstm.lstm_dim),
        )

    def _valid_actions(self, state: ParserState) -> List[int]:

        valid_actions: List[int] = []
        is_open_NT = state.is_open_NT
        num_open_NT = state.num_open_NT
        stack = state.stack_stackrnn
        buffer = state.buffer_stackrnn

        # Reduce if there are 1. the top of stack is not an NT, and
        # 2. two open NT on stack, or 3. buffer is empty
        if (is_open_NT and not is_open_NT[-1]) and (
            num_open_NT >= 2 or len(buffer) == 0
        ):
            assert len(stack) > 0
            valid_actions.append(self.reduce_idx)

        if len(buffer) > 0 and num_open_NT < self.config.max_open_NT:
            last_open_NT = None
            try:
                last_open_NT = stack.ele_from_top(is_open_NT[::-1].index(True))
            except ValueError:
                pass

            if (not self.training) or self.constraints.intent_slot_nesting:
                # if stack is empty or the last open NT is slot
                if (not last_open_NT) or last_open_NT.node in self.valid_SL_idxs:
                    valid_actions += self.valid_IN_idxs
                elif last_open_NT.node in self.valid_IN_idxs:
                    if (
                        self.constraints.no_slots_inside_unsupported
                        and state.found_unsupported
                    ):
                        pass
                    elif self.constraints.ontology:
                        valid_actions += (
                            self.ontology_constraint.valid_SL_for_IN(last_open_NT.node)
                        )
                    else:
                        valid_actions += self.valid_SL_idxs
            else:
                valid_actions += self.valid_IN_idxs
                valid_actions += self.valid_SL_idxs

        elif (not self.training) and num_open_NT >= self.config.max_open_NT:
            print(
                "not predicting NT because buffer len is "
                + str(len(buffer))
                + " and num open NTs is "
                + str(num_open_NT)
            )
        if len(buffer) > 0 and num_open_NT >= 1:
            valid_actions.append(self.shift_idx)

        assert (
            len(valid_actions) > 0
        ), "No valid actions. stack is {}, buffer is {}, num open NT: {}".format(
            str(stack), str(buffer), str(num_open_NT)
        )
        return valid_actions

    def forward(  # noqa: C901
        self, inputs: List[torch.LongTensor], beam_size=1, topk=1
    ):
        [tokens_list_rev, dict_feat_ids, dict_feat_wts, dict_feat_lengths] = (
            inputs[0],
            inputs[1],
            inputs[2],
            inputs[3],
        )

        if self.training:
            assert beam_size == 1, "beam_size must be 1 during training"
            assert len(inputs) > 4
            oracle_actions_idx_rev = inputs[4]

        beam_size = max(beam_size, 1)

        tok_embeddings = self.WORDS_LOOKUP(tokens_list_rev.unsqueeze(0)).squeeze(0)
        if self.add_dict_feat:
            if all(
                [
                    dict_feat_ids.numel() > 0,
                    dict_feat_wts.numel() > 0,
                    dict_feat_lengths.numel() > 0,
                ]
            ):
                # Add batch dimension because DictEmbedding.forward() expects.
                dict_embeddings = self.DICT_FEAT_LOOKUP(
                    tokens_list_rev.unsqueeze(0),
                    dict_feat_ids.unsqueeze(0),
                    dict_feat_wts.unsqueeze(0),
                    dict_feat_lengths.unsqueeze(0),
                ).squeeze(0)
            else:
                dict_embeddings = Variable(
                    torch.zeros(
                        [tok_embeddings.size()[0], self.DICT_FEAT_LOOKUP.embedding_dim],
                        dtype=torch.float32,
                    ),
                    requires_grad=False,
                )  # Don't use dict embeddings
            tok_embeddings = torch.cat([tok_embeddings, dict_embeddings], dim=1)

        initial_state = ParserState(self)

        for i in range(tok_embeddings.size()[0]):
            tok_embedding = tok_embeddings[i].unsqueeze(0)
            tok = tokens_list_rev[i]
            initial_state.buffer_stackrnn.push(tok_embedding, Element(tok))

        beam = [initial_state]

        while beam and any(not state.finished() for state in beam):
            # Stores plans for expansion as (score, state, action)
            plans: List[Tuple[float, ParserState, int]] = []
            # Expand current beam states
            for state in beam:
                # Keep terminal states
                if state.finished():
                    plans.append((state.neg_prob, state, -1))
                    continue

                #  translating Expression p_t = affine_transform({pbias, S,
                #  stack_summary, B, buffer_summary, A, action_summary});
                stack_summary = state.stack_stackrnn.embedding()
                action_summary = state.action_stackrnn.embedding()
                buffer_summary = state.buffer_stackrnn.embedding()
                if self.config.dropout > 0:
                    stack_summary = self.dropout_layer(stack_summary)
                    action_summary = self.dropout_layer(action_summary)
                    buffer_summary = self.dropout_layer(buffer_summary)

                # TODO: should have explicit features for # of open non-terminals
                action_p_bitvec = (
                    (self.config.ablation.use_buffer << 2)
                    | (self.config.ablation.use_stack << 1)
                    | (self.config.ablation.use_action << 0)
                )

                if action_p_bitvec == 1:
                    action_p = self.action_linear(action_summary)
                elif action_p_bitvec == 2:
                    action_p = self.action_linear(stack_summary)
                elif action_p_bitvec == 3:
                    action_p = self.action_linear(
                        torch.cat([stack_summary, action_summary], dim=1)
                    )
                elif action_p_bitvec == 4:
                    action_p = self.action_linear(buffer_summary)
                elif action_p_bitvec == 5:
                    action_p = self.action_linear(
                        torch.cat([buffer_summary, action_summary], dim=1)
                    )
                elif action_p_bitvec == 6:
                    action_p = self.action_linear(
                        torch.cat([buffer_summary, stack_summary], dim=1)
                    )
                elif action_p_bitvec == 7:
                    action_p = self.action_linear(
                        torch.cat(
                            [buffer_summary, stack_summary, action_summary], dim=1
                        )
                    )
                else:
                    raise ValueError(
                        "Need atleast one of the Parser ablation flags to be True"
                    )

                log_probs = F.log_softmax(action_p, dim=1)[0]

                for action in self._valid_actions(state):
                    plans.append((state.neg_prob - log_probs[action], state, action))

            beam = []
            # Take actions to regenerate the beam
            for neg_prob, state, predicted_action_idx in sorted(plans)[:beam_size]:
                # Skip terminal states
                if state.finished():
                    beam.append(state)
                    continue

                # Only branch out states when needed
                if beam_size > 1:
                    state = state.copy()

                state.predicted_actions_idx.append(predicted_action_idx)

                target_action_idx = predicted_action_idx
                if self.training:
                    target_action_idx = oracle_actions_idx_rev[-1]
                    oracle_actions_idx_rev = oracle_actions_idx_rev[:-1]

                if (
                    self.constraints.ignore_loss_for_unsupported
                    and state.found_unsupported
                ):
                    pass
                else:
                    state.action_scores.append(action_p)

                action_embedding = self.ACTIONS_LOOKUP(
                    Variable(torch.LongTensor([target_action_idx]))
                )
                state.action_stackrnn.push(action_embedding, Element(target_action_idx))

                if target_action_idx == self.shift_idx:
                    state.is_open_NT.append(False)
                    tok_embedding, token = state.buffer_stackrnn.pop()
                    state.stack_stackrnn.push(tok_embedding, Element(token))
                elif target_action_idx == self.reduce_idx:
                    state.num_open_NT -= 1
                    popped_rep = []
                    nt_tree = []

                    while not state.is_open_NT[-1]:
                        assert len(state.stack_stackrnn) > 0, "How come stack is empty!"
                        state.is_open_NT.pop()
                        top_of_stack = state.stack_stackrnn.pop()
                        popped_rep.append(top_of_stack[0])
                        nt_tree.append(top_of_stack[1])

                    # pop the open NT and close it
                    top_of_stack = state.stack_stackrnn.pop()
                    popped_rep.append(top_of_stack[0])
                    nt_tree.append(top_of_stack[1])

                    state.is_open_NT.pop()
                    state.is_open_NT.append(False)

                    compostional_rep = self.p_compositional(popped_rep)
                    combinedElement = Element(nt_tree)

                    state.stack_stackrnn.push(compostional_rep, combinedElement)
                elif target_action_idx in self.valid_NT_idxs:

                    # if this is root prediction and if that root is one
                    # of the unsupported intents
                    if (
                        len(state.predicted_actions_idx) == 1
                        and target_action_idx in self.ignore_subNTs_roots
                    ):
                        state.found_unsupported = True

                    state.is_open_NT.append(True)
                    state.num_open_NT += 1
                    state.stack_stackrnn.push(
                        action_embedding, Element(target_action_idx)
                    )
                else:
                    assert "not a valid action: {}".format(
                        self.actions_bidict.value(target_action_idx)
                    )

                state.neg_prob = neg_prob
                beam.append(state)
            # End for
        # End while
        assert len(beam) > 0, "How come beam is empty?"
        assert len(state.stack_stackrnn) == 1, "How come stack len is " + str(
            len(state.stack_stackrnn)
        )
        assert len(state.buffer_stackrnn) == 0, "How come buffer len is " + str(
            len(state.buffer_stackrnn)
        )

        if topk <= 1:
            state = min(beam)
            return (
                torch.LongTensor(state.predicted_actions_idx),
                torch.cat(state.action_scores),
            )
        else:
            return [
                (
                    torch.LongTensor(state.predicted_actions_idx),
                    torch.cat(state.action_scores),
                )
                for state in sorted(beam)[:topk]
            ]

    def _get_dict_feat_config(self, config: PyTextConfig, dict_embed_num: int):
        return (
            DictEmbeddingConfig(
                dict_embed_num,
                config.jobspec.features.dict_feat.embed_dim,
                config.jobspec.features.dict_feat.pooling,
            )
            if dict_embed_num > 0
            else None
        )

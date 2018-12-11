#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
import pytext.utils.cuda_utils as cuda_utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytext.config import ConfigBase
from pytext.config.component import Component, ComponentType
from pytext.data import CommonMetadata
from pytext.models import Model
from pytext.models.embeddings import EmbeddingList
from pytext.models.representations.bilstm import BiLSTM
from pytext.models.semantic_parsers.rnng.rnng_data_structures import (
    CompositionalNN,
    CompositionalSummationNN,
    CompositionFunction,
    Element,
    ParserState,
)


class CompositionalType(Enum):
    """Whether to use summation of the vectors or a BiLSTM based composition to
     generate embedding for a subtree"""

    BLSTM = "blstm"
    SUM = "sum"


class AblationParams(ConfigBase):
    """Different ablation parameters.
    use_last_open_NT_feature uses the last open non-terminal as a 1-hot feature
    when computing representation for the action classifier

    Attributes:
    use_buffer: whether to use the buffer LSTM
    use_stack: whether to use the stack LSTM
    use_action: whether to use the action LSTM
    """

    use_buffer: bool = True
    use_stack: bool = True
    use_action: bool = True
    use_last_open_NT_feature: bool = False


class RNNGConstraints(ConfigBase):
    """Constraints when computing valid actions.

    intent_slot_nesting: for the intent slot models, the top level non-terminal
    has to be an intent, an intent can only have slot non-terminals as
    children and vice-versa.

    ignore_loss_for_unsupported and no_slots_inside_unsupported:
    if the data has "unsupported" label, that is if the label has a substring
    "unsupported" in it, these flags control whether or not to compute loss or
    predict slots in side the unsupported intent
    """

    intent_slot_nesting: bool = True
    ignore_loss_for_unsupported: bool = False
    no_slots_inside_unsupported: bool = True


class RNNGParser(Model, Component):
    """
    The Recurrent Neural Network Grammar (RNNG) parser from
    Dyer et al.: https://arxiv.org/abs/1602.07776 and
    Gupta et al.: https://arxiv.org/abs/1810.07942d.
    RNNG is a neural constituency parsing algorithm that
    explicitly models compositional structure of a sentence.
    It is able to learn about hierarchical relationship among the words and
    phrases in a given sentence thereby learning the underlying tree structure.
    The paper proposes generative as well as discriminative approaches.
    In PyText we have implemented the discriminative approach for modeling
    intent slot models.
    It is a top-down shift-reduce parser than can output
    trees with non-terminals (intent and slot labels) and terminals (tokens)
    """

    __COMPONENT_TYPE__ = ComponentType.MODEL

    class Config(ConfigBase):
        version: int = 0
        lstm: BiLSTM.Config = BiLSTM.Config()
        ablation: AblationParams = AblationParams()
        constraints: RNNGConstraints = RNNGConstraints()
        max_open_NT: int = 10
        dropout: float = 0.1
        compositional_type: CompositionalType = CompositionalType.BLSTM

    @classmethod
    def from_config(cls, model_config, feature_config, metadata: CommonMetadata):
        if model_config.compositional_type == CompositionalType.SUM:
            p_compositional = CompositionalSummationNN(
                lstm_dim=model_config.lstm.lstm_dim
            )
        elif model_config.compositional_type == CompositionalType.BLSTM:
            p_compositional = CompositionalNN(lstm_dim=model_config.lstm.lstm_dim)
        else:
            raise ValueError(
                "Cannot understand compositional flag {}".format(
                    model_config.compositional_type
                )
            )

        return cls(
            ablation=model_config.ablation,
            constraints=model_config.constraints,
            lstm_num_layers=model_config.lstm.num_layers,
            lstm_dim=model_config.lstm.lstm_dim,
            max_open_NT=model_config.max_open_NT,
            dropout=model_config.dropout,
            actions_vocab=metadata.actions_vocab,
            shift_idx=metadata.shift_idx,
            reduce_idx=metadata.reduce_idx,
            ignore_subNTs_roots=metadata.ignore_subNTs_roots,
            valid_NT_idxs=metadata.valid_NT_idxs,
            valid_IN_idxs=metadata.valid_IN_idxs,
            valid_SL_idxs=metadata.valid_SL_idxs,
            embedding=Model.create_embedding(feature_config, metadata=metadata),
            p_compositional=p_compositional,
        )

    def __init__(
        self,
        ablation: AblationParams,
        constraints: RNNGConstraints,
        lstm_num_layers: int,
        lstm_dim: int,
        max_open_NT: int,
        dropout: float,
        actions_vocab,
        shift_idx: int,
        reduce_idx: int,
        ignore_subNTs_roots: List[int],
        valid_NT_idxs: List[int],
        valid_IN_idxs: List[int],
        valid_SL_idxs: List[int],
        embedding: EmbeddingList,
        p_compositional: CompositionFunction,
    ) -> None:
        """
        Initialize the model

        Args:
        ablation : AblationParams
            Features/RNNs to use
        constraints : RNNGConstraints
            Constraints to use when computing valid actions
        lstm_num_layers : int
            number of layers in the LSTMs
        lstm_dim : int
            size of LSTM
        max_open_NT : int
            number of maximum open non-terminals allowed on the stack.
            After that, the only valid actions are SHIFT and REDUCE
        dropout : float
            dropout parameter
        actions_vocab : Vocab (right now torchtext.vocab.Vocab)
            dictionary of actions
        shift_idx : int
            index of shift action
        reduce_idx : int
            index of reduce action
        ignore_subNTs_roots : List[int]
            for these top non-terminals, ignore loss for all subsequent actions
        valid_NT_idxs : List[int]
            indices of all non-terminals
        valid_IN_idxs : List[int]
            indices of intent non-terminals
        valid_SL_idxs : List[int]
            indices of slot non-terminals
        embedding : EmbeddingList
            embeddings for the tokens
        p_compositional : CompositionFunction
            Composition function to use to get embedding of a sub-tree


        Returns:
        None


        """

        nn.Module.__init__(self)

        self.embedding = embedding
        # self.embedding.config: FeatureConfig object cannot be pickled but,
        # we require the model to be pickled for passing from one worker process
        # for Hogwild training. Hence, setting the config to None
        self.embedding.config = None

        self.p_compositional = p_compositional
        self.ablation_use_last_open_NT_feature = ablation.use_last_open_NT_feature
        self.ablation_use_buffer = ablation.use_buffer
        self.ablation_use_stack = ablation.use_stack
        self.ablation_use_action = ablation.use_action

        self.constraints_intent_slot_nesting = constraints.intent_slot_nesting
        self.constraints_no_slots_inside_unsupported = (
            constraints.no_slots_inside_unsupported
        )
        self.constraints_ignore_loss_for_unsupported = (
            constraints.ignore_loss_for_unsupported
        )

        self.lstm_num_layers = lstm_num_layers
        self.lstm_dim = lstm_dim
        self.max_open_NT = max_open_NT
        self.actions_vocab = actions_vocab
        self.shift_idx = shift_idx
        self.reduce_idx = reduce_idx
        self.ignore_subNTs_roots = ignore_subNTs_roots
        self.valid_NT_idxs = valid_NT_idxs
        self.valid_IN_idxs = valid_IN_idxs
        self.valid_SL_idxs = valid_SL_idxs

        num_actions = len(actions_vocab)
        lstm_count = ablation.use_buffer + ablation.use_stack + ablation.use_action
        if lstm_count == 0:
            raise ValueError("Need atleast one of the LSTMs to be true")

        self.action_linear = nn.Sequential(
            nn.Linear(
                lstm_count * lstm_dim + num_actions * ablation.use_last_open_NT_feature,
                lstm_dim,
            ),
            nn.ReLU(),
            nn.Linear(lstm_dim, num_actions),
        )
        self.dropout_layer = nn.Dropout(p=dropout)
        self.buff_rnn = nn.LSTM(
            embedding.embedding_dim,
            lstm_dim,
            num_layers=lstm_num_layers,
            dropout=dropout,
        )
        self.stack_rnn = nn.LSTM(
            lstm_dim, lstm_dim, num_layers=lstm_num_layers, dropout=dropout
        )
        self.action_rnn = nn.LSTM(
            lstm_dim, lstm_dim, num_layers=lstm_num_layers, dropout=dropout
        )

        self.pempty_buffer_emb = nn.Parameter(torch.randn(1, lstm_dim))
        self.empty_stack_emb = nn.Parameter(torch.randn(1, lstm_dim))
        self.empty_action_emb = nn.Parameter(torch.randn(1, lstm_dim))

        self.actions_lookup = nn.Embedding(num_actions, lstm_dim)
        self.loss_func = nn.CrossEntropyLoss()

    def forward(
        self,
        tokens: torch.Tensor,
        seq_lens: torch.Tensor,
        dict_feat: Optional[Tuple[torch.Tensor, ...]] = None,
        actions: Optional[List[List[int]]] = None,
        beam_size: int = 1,
        topk: int = 1,
    ):
        """RNNG forward function.

        Args:
        tokens : torch.Tensor
            list of tokens
        seq_lens : torch.Tensor
            list of sequence lengths
        dict_feat : Optional[Tuple[torch.Tensor, ...]]
            dictionary or gazetter features for each token
        actions : Optional[List[List[int]]]
            Used only during training. Oracle actions for the instances.
        beam_size : int
            Beam size; used only during inference
        topk : int
            Number of top results from the method. if beam_size is 1 this is 1.


        Returns:
        if topk == 1
            tuple of list of predicted actions and list of corresponding scores
        else
            returns
            list of tuple of list of predicted actions and list of corresponding scores


        """
        if self.training:
            assert beam_size == 1, "beam_size must be 1 during training"
            assert actions is not None, "actions must be provided for training"
            actions_idx_rev = list(reversed(actions[0]))
        else:
            torch.manual_seed(0)

        beam_size = max(beam_size, 1)

        # Reverse the order of indices along last axis before embedding lookup.
        tokens_list_rev = torch.flip(tokens, [len(tokens.size()) - 1])
        dict_feat_rev = None
        if dict_feat:
            dict_ids, dict_weights, dict_lengths = dict_feat
            dict_ids_rev = torch.flip(dict_ids, [len(dict_ids.size()) - 1])
            dict_weights_rev = torch.flip(dict_weights, [len(dict_weights.size()) - 1])
            dict_lengths_rev = torch.flip(dict_lengths, [len(dict_lengths.size()) - 1])
            dict_feat_rev = (dict_ids_rev, dict_weights_rev, dict_lengths_rev)

        embedding_input = (
            [tokens_list_rev, dict_feat_rev]
            if dict_feat_rev is not None
            else [tokens_list_rev]
        )
        tok_embeddings = self.embedding(*embedding_input)

        # Batch size is always = 1. So we squeeze the batch_size dimension.
        tok_embeddings = tok_embeddings.squeeze(0)
        tokens_list_rev = tokens_list_rev.squeeze(0)

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
                stack = state.stack_stackrnn
                stack_summary = stack.embedding()
                action_summary = state.action_stackrnn.embedding()
                buffer_summary = state.buffer_stackrnn.embedding()
                if self.dropout_layer.p > 0:
                    stack_summary = self.dropout_layer(stack_summary)
                    action_summary = self.dropout_layer(action_summary)
                    buffer_summary = self.dropout_layer(buffer_summary)

                # feature for index of last open non-terminal
                last_open_NT_feature = torch.zeros(len(self.actions_vocab))
                open_NT_exists = state.num_open_NT > 0

                if (
                    len(stack) > 0
                    and open_NT_exists
                    and self.ablation_use_last_open_NT_feature
                ):
                    last_open_NT = None
                    try:
                        open_NT = state.is_open_NT[::-1].index(True)
                        last_open_NT = stack.ele_from_top(open_NT)
                    except ValueError:
                        pass
                    if last_open_NT:
                        last_open_NT_feature[last_open_NT.node] = 1.0
                last_open_NT_feature = last_open_NT_feature.unsqueeze(0)

                summaries = []
                if self.ablation_use_buffer:
                    summaries.append(buffer_summary)
                if self.ablation_use_stack:
                    summaries.append(stack_summary)
                if self.ablation_use_action:
                    summaries.append(action_summary)
                if self.ablation_use_last_open_NT_feature:
                    summaries.append(last_open_NT_feature)

                action_p = self.action_linear(torch.cat(summaries, dim=1))

                log_probs = F.log_softmax(action_p, dim=1)[0]

                for action in self.valid_actions(state):
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
                    assert (
                        len(actions_idx_rev) > 0
                    ), "Actions and tokens may not be in sync."
                    target_action_idx = actions_idx_rev[-1]
                    actions_idx_rev = actions_idx_rev[:-1]

                if (
                    self.constraints_ignore_loss_for_unsupported
                    and state.found_unsupported
                ):
                    pass
                else:
                    state.action_scores.append(action_p)

                action_embedding = self.actions_lookup(
                    cuda_utils.Variable(torch.LongTensor([target_action_idx]))
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
                        self.actions_vocab.itos[target_action_idx]
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

        # Add batch dimension before returning.
        if topk <= 1:
            state = min(beam)
            return (
                torch.LongTensor(state.predicted_actions_idx).unsqueeze(0),
                torch.cat(state.action_scores).unsqueeze(0),
            )
        else:
            return [
                (
                    torch.LongTensor(state.predicted_actions_idx).unsqueeze(0),
                    torch.cat(state.action_scores).unsqueeze(0),
                )
                for state in sorted(beam)[:topk]
            ]

    # Initializes LSTM parameters
    def init_lstm(self):
        return (
            torch.zeros(self.lstm_num_layers, 1, self.lstm_dim),
            torch.zeros(self.lstm_num_layers, 1, self.lstm_dim),
        )

    def valid_actions(self, state: ParserState) -> List[int]:
        """Used for restricting the set of possible action predictions

        Args:
        state : ParserState
            The state of the stack, buffer and action

        Returns:
        List[int]
            indices of the valid actions

        """
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

        if len(buffer) > 0 and num_open_NT < self.max_open_NT:
            last_open_NT = None
            try:
                last_open_NT = stack.ele_from_top(is_open_NT[::-1].index(True))
            except ValueError:
                pass

            if (not self.training) or self.constraints_intent_slot_nesting:
                # if stack is empty or the last open NT is slot
                if (not last_open_NT) or last_open_NT.node in self.valid_SL_idxs:
                    valid_actions += self.valid_IN_idxs
                elif last_open_NT.node in self.valid_IN_idxs:
                    if (
                        self.constraints_no_slots_inside_unsupported
                        and state.found_unsupported
                    ):
                        pass
                    else:
                        valid_actions += self.valid_SL_idxs
            else:
                valid_actions += self.valid_IN_idxs
                valid_actions += self.valid_SL_idxs

        elif (not self.training) and num_open_NT >= self.max_open_NT:
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

    def get_param_groups_for_optimizer(self):
        """
        This is called by code that looks for an instance of pytext.models.model.Model.
        """
        return [{"params": self.parameters()}]

    def get_loss(
        self, logits: torch.Tensor, target_actions: torch.Tensor, context: torch.Tensor
    ):
        # action scores is a 2D Tensor of dims sequence_length x number_of_actions
        # targets is a 1D list of integers of length sequence_length

        # Get rid of the batch dimension
        action_scores = logits[1].squeeze(0)
        target_actions = target_actions.squeeze(0)

        action_scores_list = torch.chunk(action_scores, action_scores.size()[0])
        target_vars = [
            cuda_utils.Variable(torch.LongTensor([t])) for t in target_actions
        ]
        losses = [
            self.loss_func(action, target).view(1)
            for action, target in zip(action_scores_list, target_vars)
        ]
        total_loss = torch.sum(torch.cat(losses)) if len(losses) > 0 else None
        return total_loss

    def get_pred(self, logits: Tuple[torch.Tensor, torch.Tensor], *args):
        predicted_action_idx, predicted_action_scores = logits
        predicted_scores = [
            np.exp(np.max(action_scores)).item() / np.sum(np.exp(action_scores)).item()
            for action_scores in predicted_action_scores.detach().squeeze(0).tolist()
        ]

        return predicted_action_idx.tolist(), [predicted_scores]

    def save_modules(self, *args, **kwargs):
        pass

    def contextualize(self, context):
        self.context = context

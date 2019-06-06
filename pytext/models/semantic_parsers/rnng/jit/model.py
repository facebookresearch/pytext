#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List, Tuple

import numpy as np
import pytext.models.semantic_parsers.rnng.rnng_constant as const
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytext.config import ConfigBase
from pytext.data import CommonMetadata
from pytext.models import BaseModel, Model
from pytext.models.representations.bilstm import BiLSTM
from pytext.models.semantic_parsers.rnng.rnng_data_structures import (
    CompositionalNN,
    CompositionalSummationNN,
)
from pytext.models.semantic_parsers.rnng.rnng_parser import RNNGParser
from pytext.utils import cuda
from pytext.utils.torch import list_membership
from torch import jit


@jit.script
class LSTMStateStack:
    def __init__(self, num_layers: int, hidden_size: int, device: str):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device
        self.last_state = (
            torch.zeros([num_layers, const.BATCH_SIZE, hidden_size], device=device),
            torch.zeros([num_layers, const.BATCH_SIZE, hidden_size], device=device),
        )
        # Stack of hidden state, element
        self.stack = [(self.last_state[0][-1], const.ROOT_ELEMENT)]

    def push(self, new_state: Tuple[torch.Tensor, torch.Tensor], element: int) -> None:
        self.last_state = new_state
        self.stack.append((self.last_state[0][-1], element))

    def embedding(self) -> torch.Tensor:
        return self.stack[-1][0]

    def element_from_top(self, index: int) -> int:
        idx = -(index + 1)
        assert idx >= 0, "idx should not be less than 0!"
        return self.stack[-(index + 1)][1]

    def pop(self) -> Tuple[torch.Tensor, int]:
        return self.stack.pop()

    def size(self) -> int:
        return len(self.stack) - 1

    def copy(self):
        other = LSTMStateStack(self.num_layers, self.hidden_size, self.device)
        other.stack = list(self.stack)
        return other


@jit.script
class ParserState:
    """
    Maintains state of the Parser. Useful for beam search
    """

    def __init__(
        self,
        buffer_stack: LSTMStateStack,
        stack_stack: LSTMStateStack,
        action_stack: LSTMStateStack,
    ):
        self.buffer_state_stack = buffer_stack
        self.stack_state_stack = stack_stack
        self.action_state_stack = action_stack

        self.predicted_actions_idx = jit.annotate(List[int], [])
        self.action_scores = []

        self.is_open_NT = jit.annotate(List[bool], [])
        self.open_NT = jit.annotate(List[int], [])
        self.found_unsupported = False
        # dummy tensor as place holder
        self.action_p = torch.zeros(1)

        # negative cumulative log prob so sort(states) is in descending order
        self.neg_prob = 0.0

    def finished(self):
        return (
            self.stack_state_stack.size() == 1 and self.buffer_state_stack.size() == 0
        )

    def copy(self):
        other = ParserState(
            self.buffer_state_stack.copy(),
            self.stack_state_stack.copy(),
            self.action_state_stack.copy(),
        )
        other.predicted_actions_idx = self.predicted_actions_idx.copy()
        other.action_scores = self.action_scores.copy()
        other.is_open_NT = self.is_open_NT.copy()
        other.open_NT = self.open_NT.copy()
        other.neg_prob = self.neg_prob
        other.found_unsupported = self.found_unsupported
        # detach to avoid making copies, only called in inference to share data
        other.action_p = self.action_p.detach()
        return other

    def __lt__(self, other):
        # type: (ParserState) -> bool
        return self.neg_prob < other.neg_prob


@jit.script
class Plan:
    def __init__(self, score: float, action: int, state: ParserState):
        self.score = score
        self.action = action
        self.state = state

    def __lt__(self, other):
        # type: (Plan) -> bool
        return self.score < other.score


class RNNGParserJIT(jit.ScriptModule):
    __constants__ = [
        "lstm_num_layers",
        "lstm_dim",
        "dropout",
        "num_actions",
        "shift_idx",
        "reduce_idx",
        "max_open_NT",
        "ablation_use_stack",
        "ablation_use_buffer",
        "ablation_use_action",
        "ablation_use_last_open_NT_feature",
        "valid_NT_idxs",
        "valid_IN_idxs",
        "valid_SL_idxs",
        "constraints_intent_slot_nesting",
        "constraints_no_slots_inside_unsupported",
        "constraints_ignore_loss_for_unsupported",
        "ignore_subNTs_roots",
        "action_linear",
        "embedding_dim",
        "device",
    ]

    def __init__(
        self,
        ablation: RNNGParser.Config.AblationParams,
        constraints: RNNGParser.Config.RNNGConstraints,
        lstm_num_layers: int,
        lstm_dim: int,
        max_open_NT: int,
        dropout: float,
        num_actions: int,
        shift_idx: int,
        reduce_idx: int,
        ignore_subNTs_roots: List[int],
        valid_NT_idxs: List[int],
        valid_IN_idxs: List[int],
        valid_SL_idxs: List[int],
        embedding,
        embedding_dim: int,
        p_compositional,
        device: str,
    ) -> None:
        super().__init__()
        self.device = device
        self.embedding = embedding
        self.lstm_num_layers = lstm_num_layers
        self.lstm_dim = lstm_dim

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
        self.max_open_NT = max_open_NT
        self.shift_idx = shift_idx
        self.reduce_idx = reduce_idx
        self.ignore_subNTs_roots = ignore_subNTs_roots
        self.valid_NT_idxs = valid_NT_idxs
        self.valid_IN_idxs = valid_IN_idxs
        self.valid_SL_idxs = valid_SL_idxs

        self.num_actions = num_actions
        lstm_count = ablation.use_buffer + ablation.use_stack + ablation.use_action
        if lstm_count == 0:
            raise ValueError("Need at least one of the LSTMs to be true")

        self.action_linear = nn.Sequential(
            nn.Linear(
                lstm_count * lstm_dim
                + self.num_actions * ablation.use_last_open_NT_feature,
                lstm_dim,
            ),
            nn.ReLU(),
            nn.Linear(lstm_dim, self.num_actions),
        )
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(p=dropout)
        self.buffer_lstm = nn.LSTM(
            embedding_dim,
            lstm_dim,
            num_layers=lstm_num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.stack_lstm = nn.LSTM(
            lstm_dim, lstm_dim, num_layers=lstm_num_layers, dropout=dropout
        )
        self.action_lstm = nn.LSTM(
            lstm_dim, lstm_dim, num_layers=lstm_num_layers, dropout=dropout
        )

        self.actions_lookup = nn.Embedding(self.num_actions, lstm_dim)
        self.loss_func = nn.CrossEntropyLoss()

    @jit.script_method
    def forward(
        self,
        tokens: torch.Tensor,
        seq_lens: torch.Tensor,
        dict_feat: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        actions: List[List[int]],
        contextual_token_embeddings: torch.Tensor,
        beam_size: int = 1,
        top_k: int = 1,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:

        actions_idx = jit.annotate(List[int], [])
        if self.training:
            # batch size is only 1 for now
            actions_idx = actions[0]
            assert len(actions_idx) > 0, "actions must be provided for training"
        else:
            torch.manual_seed(0)
        token_embeddings = self.embedding(
            tokens, dict_feat, contextual_token_embeddings
        )
        beam = [self.gen_init_state(tokens, token_embeddings)]
        all_finished = False
        while not all_finished:
            # Stores plans for expansion as (score, state, action)
            plans = jit.annotate(List[Plan], [])
            all_finished = True
            # Expand current beam states
            for state in beam:
                # Keep terminal states
                if state.finished():
                    plans.append(Plan(state.neg_prob, const.TERMINAL_ELEMENT, state))
                else:
                    all_finished = False
                    plans.extend(self.gen_plans(state))

            beam.clear()
            # Take actions to regenerate the beam
            plans.sort()
            for plan in plans[:beam_size]:
                beam.append(self.execute_plan(plan, actions_idx, beam_size))

        # sanity check
        assert len(beam) > 0, "How come beam is empty?"

        beam.sort()
        res = jit.annotate(List[Tuple[torch.Tensor, torch.Tensor]], [])
        for state in beam[:top_k]:
            res.append(
                (
                    torch.tensor([state.predicted_actions_idx]),
                    # Unsqueeze to add batch dimension
                    torch.cat(state.action_scores).unsqueeze(0),
                )
            )
        return res

    @jit.script_method
    def gen_init_state(self, tokens, token_embeddings):
        token_embeddings = torch.flip(token_embeddings, [len(tokens.size()) - 1])
        # Reverse the order of input tokens.
        tokens_list_rev = torch.flip(tokens, [len(tokens.size()) - 1])
        # Batch size is always = 1. So we squeeze the batch_size dimension.
        token_embeddings = token_embeddings.squeeze(0)
        tokens_list_rev = tokens_list_rev.squeeze(0)

        buffer_stack = LSTMStateStack(self.lstm_num_layers, self.lstm_dim, self.device)
        for i in range(token_embeddings.size()[0]):
            embedding = token_embeddings[i].unsqueeze(0).unsqueeze(0)
            _, new_state = self.buffer_lstm(embedding, buffer_stack.last_state)
            buffer_stack.push(new_state, tokens_list_rev[i])

        return ParserState(
            buffer_stack,
            LSTMStateStack(self.lstm_num_layers, self.lstm_dim, self.device),
            LSTMStateStack(self.lstm_num_layers, self.lstm_dim, self.device),
        )

    @jit.script_method
    def get_summary(self, state_stack: LSTMStateStack):
        summary = state_stack.embedding()
        if self.dropout > 0:
            self.dropout_layer(summary)
        return summary

    @jit.script_method
    def gen_plans(self, state: ParserState):
        plans = jit.annotate(List[Plan], [])
        # translating Expression p_t = affine_transform({pbias, S, stack_summary,
        # B, buffer_summary, A, action_summary});
        # list comprehension with ifs not supported by jit yet
        summaries = []
        for stack_tuple in (
            (state.stack_state_stack, self.ablation_use_stack),
            (state.buffer_state_stack, self.ablation_use_buffer),
            (state.action_state_stack, self.ablation_use_action),
        ):
            stack, flag = stack_tuple
            if flag:
                summaries.append(self.get_summary(stack))

        if self.ablation_use_last_open_NT_feature:
            # feature for index of last open non-terminal
            last_open_NT_feature = torch.zeros(self.num_actions)
            if len(state.open_NT) > 0:
                last_open_NT_feature[state.open_NT[-1]] = 1.0
            summaries.append(last_open_NT_feature.unsqueeze(0))

        state.action_p = self.action_linear(torch.cat(summaries, dim=1))
        log_probs = F.log_softmax(state.action_p, dim=1)[0]

        for action in self.valid_actions(state):
            plans.append(
                Plan(
                    score=state.neg_prob - int(log_probs[action].item()),
                    action=action,
                    state=state,
                )
            )
        return plans

    @jit.script_method
    def execute_plan(self, plan: Plan, actions_idx: List[int], beam_size: int):
        # Skip terminal states
        state = plan.state
        if state.finished():
            return state

        # Only branch out states when needed
        if beam_size > 1:
            state = plan.state.copy()

        state.predicted_actions_idx.append(plan.action)

        target_action_idx = plan.action
        if self.training:
            actions_taken = state.action_state_stack.size()
            assert actions_taken < len(
                actions_idx
            ), "Actions and tokens may not be in sync."
            target_action_idx = actions_idx[actions_taken]

        if not (
            self.constraints_ignore_loss_for_unsupported and state.found_unsupported
        ):
            state.action_scores.append(state.action_p)

        self.apply_action(state, target_action_idx)

        state.neg_prob = plan.score
        return state

    @jit.script_method
    def push_action(self, embedding, action_idx: int, stack: LSTMStateStack):
        # unsqueeze to set the seq_len dim to 1
        _, new_state = self.action_lstm(embedding.unsqueeze(0), stack.last_state)
        stack.push(new_state, action_idx)

    @jit.script_method
    def push_stack(self, embedding, element: int, is_open_nt: bool, state: ParserState):
        stack = state.stack_state_stack
        # unsqueeze to set the seq_len dim to 1
        _, new_state = self.stack_lstm(embedding.unsqueeze(0), stack.last_state)
        stack.push(new_state, element)
        state.is_open_NT.append(is_open_nt)
        if is_open_nt:
            state.open_NT.append(element)

    @jit.script_method
    def pop_stack(self, state: ParserState):
        if state.is_open_NT.pop():
            state.open_NT.pop()
        return state.stack_state_stack.pop()[0]

    @jit.script_method
    def apply_action(self, state: ParserState, target_action_idx: int) -> None:
        action_t = torch.tensor([target_action_idx])
        # action_t.requires_grad_()
        action_embedding = self.actions_lookup(action_t)
        self.push_action(action_embedding, target_action_idx, state.action_state_stack)

        # Update stack_state_stack
        if target_action_idx == self.shift_idx:
            # To SHIFT,
            # 1. Pop T from buffer
            # 2. Push T into stack
            token_embedding, token_idx = state.buffer_state_stack.pop()
            self.push_stack(token_embedding, token_idx, False, state)

        elif target_action_idx == self.reduce_idx:
            # To REDUCE
            # 1. Pop Ts from stack until hit NT
            # 2. Pop the open NT from stack and close it
            # 3. Compute compositionalRep and push into stack
            popped_rep = []
            while not state.is_open_NT[-1]:
                popped_rep.append(self.pop_stack(state))

            # pop the open NT and close it
            popped_rep.append(self.pop_stack(state))
            compostional_rep = self.p_compositional(popped_rep)
            self.push_stack(compostional_rep, const.TREE_ELEMENT, False, state)

        elif list_membership(target_action_idx, self.valid_NT_idxs):

            # if this is root prediction and if that root is one
            # of the unsupported intents
            if len(state.predicted_actions_idx) == 1 and list_membership(
                target_action_idx, self.ignore_subNTs_roots
            ):
                state.found_unsupported = True
            self.push_stack(action_embedding, target_action_idx, True, state)
        else:
            assert False, "not a valid action: {}".format(target_action_idx)

    @jit.script_method
    def valid_actions(self, state: ParserState) -> List[int]:

        valid_actions = jit.annotate(List[int], [])
        is_open_NT = state.is_open_NT
        num_open_NT = len(state.open_NT)
        stack = state.stack_state_stack
        buffer = state.buffer_state_stack

        # Can REDUCE if
        # 1. Top of multi-element stack is not an NT, and
        # 2. Two open NTs on stack, or buffer is empty
        if (
            len(is_open_NT) > 0 and not is_open_NT[-1] and not len(is_open_NT) == 1
        ) and (num_open_NT >= 2 or buffer.size() == 0):
            assert stack.size() > 0
            valid_actions.append(self.reduce_idx)

        if buffer.size() > 0 and num_open_NT < self.max_open_NT:
            if (not self.training) or self.constraints_intent_slot_nesting:
                # if stack is empty or the last open NT is slot
                if (len(state.open_NT) == 0) or list_membership(
                    state.open_NT[-1], self.valid_SL_idxs
                ):
                    valid_actions += self.valid_IN_idxs
                elif list_membership(state.open_NT[-1], self.valid_IN_idxs):
                    if not (
                        self.constraints_no_slots_inside_unsupported
                        and state.found_unsupported
                    ):
                        valid_actions += self.valid_SL_idxs
            else:
                valid_actions.extend(self.valid_IN_idxs)
                valid_actions.extend(self.valid_SL_idxs)

        elif (not self.training) and num_open_NT >= self.max_open_NT:
            print(
                "not predicting NT, buffer len is {}, num open NTs is {}".format(
                    buffer.size(), num_open_NT
                )
            )

        # Can SHIFT if
        # 1. Buffer is non-empty, and
        # 2. At least one open NT on stack
        if buffer.size() > 0 and num_open_NT >= 1:
            valid_actions.append(self.shift_idx)

        return valid_actions


class RNNGModel(BaseModel):
    class Config(ConfigBase):
        lstm: BiLSTM.Config = BiLSTM.Config()
        ablation: RNNGParser.Config.AblationParams = RNNGParser.Config.AblationParams()
        constraints: RNNGParser.Config.RNNGConstraints = (
            RNNGParser.Config.RNNGConstraints()
        )
        max_open_NT: int = 10
        dropout: float = 0.1
        compositional_type: RNNGParser.Config.CompositionalType = (
            RNNGParser.Config.CompositionalType.BLSTM
        )

    @classmethod
    def trace_embedding(cls, emb_module, contextual_emb_dim):
        dummy_input = (
            torch.tensor([[1], [1]]),
            (
                torch.tensor([[1], [1]]),
                torch.tensor([[1.5], [2.5]]),
                torch.tensor([[1], [1]]),
            ),
            torch.tensor([[1.0] * contextual_emb_dim, [1.0] * contextual_emb_dim]),
        )
        return torch.jit.trace(emb_module, dummy_input)

    @classmethod
    def from_config(cls, model_config, feature_config, metadata: CommonMetadata):
        device = (
            "cuda:{}".format(torch.cuda.current_device())
            if cuda.CUDA_ENABLED
            else "cpu"
        )
        if model_config.compositional_type == RNNGParser.Config.CompositionalType.SUM:
            p_compositional = CompositionalSummationNN(
                lstm_dim=model_config.lstm.lstm_dim
            )
        elif (
            model_config.compositional_type == RNNGParser.Config.CompositionalType.BLSTM
        ):
            p_compositional = CompositionalNN(
                lstm_dim=model_config.lstm.lstm_dim, device=device
            )
        else:
            raise ValueError(
                "Cannot understand compositional flag {}".format(
                    model_config.compositional_type
                )
            )
        emb_module = Model.create_embedding(feature_config, metadata=metadata)
        embedding = cls.trace_embedding(
            emb_module, feature_config.contextual_token_embedding.embed_dim
        )

        return cls(
            ablation=model_config.ablation,
            constraints=model_config.constraints,
            lstm_num_layers=model_config.lstm.num_layers,
            lstm_dim=model_config.lstm.lstm_dim,
            max_open_NT=model_config.max_open_NT,
            dropout=model_config.dropout,
            num_actions=len(metadata.actions_vocab),
            shift_idx=metadata.shift_idx,
            reduce_idx=metadata.reduce_idx,
            ignore_subNTs_roots=metadata.ignore_subNTs_roots,
            valid_NT_idxs=metadata.valid_NT_idxs,
            valid_IN_idxs=metadata.valid_IN_idxs,
            valid_SL_idxs=metadata.valid_SL_idxs,
            embedding=embedding,
            embedding_dim=emb_module.embedding_dim,
            p_compositional=p_compositional,
            device=device,
        )

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.jit_model = RNNGParserJIT(*args, **kwargs)
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, *args, **kwargs):
        return self.jit_model(*args, **kwargs)

    def get_loss(
        self,
        logits: List[Tuple[torch.Tensor, torch.Tensor]],
        target_actions: torch.Tensor,
        context: torch.Tensor,
    ):
        """
        Shapes:
            logits[1]: action scores: (1, action_length, number_of_actions)
            target_actions: (1, action_length)
        """
        # squeeze to get rid of the batch dimension
        # logits[0] is the top1 result
        action_scores = logits[0][1].squeeze(0)
        target_actions = target_actions[0].squeeze(0)

        action_scores_list = torch.chunk(action_scores, action_scores.size()[0])
        target_vars = torch.chunk(target_actions, target_actions.size()[0])
        losses = [
            self.loss_func(action, target).view(1)
            for action, target in zip(action_scores_list, target_vars)
        ]
        total_loss = torch.sum(torch.cat(losses)) if len(losses) > 0 else None
        return total_loss

    def get_single_pred(self, logits: Tuple[torch.Tensor, torch.Tensor], *args):
        predicted_action_idx, predicted_action_scores = logits
        predicted_scores = [
            np.exp(np.max(action_scores)).item() / np.sum(np.exp(action_scores)).item()
            for action_scores in predicted_action_scores.detach().squeeze(0).tolist()
        ]
        # remove the batch dimension since it's only 1
        return predicted_action_idx.tolist()[0], predicted_scores

    # Supports beam search by checking if top K exists return type
    def get_pred(self, logits: List[Tuple[torch.Tensor, torch.Tensor]], *args):
        """
        Return Shapes:
            preds: batch (1) * topk * action_len
            scores: batch (1) * topk * (action_len * number_of_actions)
        """
        n = len(logits)
        all_action_idx: List[List[int]] = [[]] * n
        all_scores: List[List[float]] = [[]] * n
        for i, l in enumerate(logits):
            all_action_idx[i], all_scores[i] = self.get_single_pred(l, *args)

        # add back batch dimension
        return [all_action_idx], [all_scores]

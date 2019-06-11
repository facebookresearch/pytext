#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List, Tuple

import torch
import torch.nn.functional as F
from pytext.utils.data import unkify as unk
from pytext.utils.torch import Vocabulary
from torch import jit


class RNNGInference(jit.ScriptModule):
    __constants__ = ["CLOSE_BRACKET", "OPEN_BRACKET"]
    OPEN_BRACKET = "["
    CLOSE_BRACKET = "]"

    def __init__(
        self,
        embedding,
        jit_module,
        word_vocab,
        dict_vocab,
        action_vocab,
        word_unk_idx=0,
        dict_unk_idx=0,
    ):
        super().__init__()
        self.word_vocab = Vocabulary(word_vocab.itos, unk_idx=word_unk_idx)
        self.dict_vocab = Vocabulary(dict_vocab.itos, unk_idx=dict_unk_idx)
        self.action_vocab = Vocabulary(action_vocab.itos, unk_idx=-1)
        self.embedding = embedding
        self.jit_module = jit_module

    @jit.script_method
    def unkify(self, tokens: List[str]) -> List[str]:
        word_ids = self.word_vocab.lookup_indices_1d(tokens)
        # unkify the tokens
        for i in range(len(word_ids)):
            if word_ids[i] == self.word_vocab.unk_idx:
                tokens[i] = unk(tokens[i])
        return tokens

    @jit.script_method
    def actions_to_seqlogical(self, actions, tokens: List[str]):
        token_idx = 0
        res = jit.annotate(List[str], [])
        for idx in range(actions.size(0)):
            action = int(actions[idx])
            if action == self.jit_module.reduce_idx:
                res.append(self.CLOSE_BRACKET)
            elif action == self.jit_module.shift_idx:
                res.append(tokens[token_idx])
                token_idx += 1
            else:
                res.append(self.OPEN_BRACKET + self.action_vocab.lookup_word(action))
        return res

    @jit.script_method
    def forward(
        self,
        tokens: List[str],
        dict_feat: Tuple[List[str], List[float], List[int]],
        contextual_token_embeddings: List[float],
        beam_size: int = 1,
        top_k: int = 1,
    ):
        token_ids = self.word_vocab.lookup_indices_1d(self.unkify(tokens))
        dict_tokens, dict_weights, dict_lengths = dict_feat
        dict_ids = self.dict_vocab.lookup_indices_1d(dict_tokens)
        token_ids_tensor = torch.tensor([token_ids])
        embed = self.embedding(
            token_ids_tensor,
            (
                torch.tensor([dict_ids]),
                torch.tensor([dict_weights], dtype=torch.float),
                torch.tensor([dict_lengths]),
            ),
            torch.tensor([contextual_token_embeddings], dtype=torch.float),
        )
        raw_results = self.jit_module(
            tokens=token_ids_tensor,
            token_embeddings=embed,
            actions=(),
            beam_size=beam_size,
            top_k=top_k,
        )
        results = jit.annotate(List[Tuple[List[str], List[float]]], [])
        for result in raw_results:
            actions, scores = result
            seq_logical = self.actions_to_seqlogical(actions.squeeze(0), tokens)
            normalized_scores = F.softmax(scores, 2).max(2)[0].squeeze(0)
            float_scores = jit.annotate(List[float], [])
            # TODO this can be done more efficiently once JIT provide native support
            for idx in range(normalized_scores.size(0)):
                float_scores.append(float(normalized_scores[idx]))
            results.append((seq_logical, float_scores))
        return results

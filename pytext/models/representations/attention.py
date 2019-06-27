#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytext.models.module import Module


class DotProductSelfAttention(Module):
    """
    Given vector w and token vectors = {t1, t2, ..., t_n}, compute self attention
    weights to weighs the tokens
    * a_j = softmax(w . t_j)
    """

    class Config(Module.Config):
        input_dim: int = 32

    @classmethod
    def from_config(cls, config: Config):
        return cls(config.input_dim)

    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, tokens, tokens_mask):
        """
        Input:
            x: batch_size * seq_len * input_dim
            x_mask: batch_size * seq_len (1 for padding, 0 for true)
        Output:
            alpha: batch_size * seq_len
        """
        scores = self.linear(tokens).squeeze(2)
        scores.data.masked_fill_(tokens_mask.data, -float("inf"))
        return F.softmax(scores, dim=-1)


class SequenceAlignedAttention(Module):
    """
    Given sequences P and Q, computes attention weights for each element in P by
    matching Q with each element in P.
    * a_i_j = softmax(p_i . q_j) where softmax is computed by summing over q_j
    """

    class Config(Module.Config):
        proj_dim: int = 32

    @classmethod
    def from_config(cls, config: Config):
        return cls(config.proj_dim)

    def __init__(self, proj_dim):
        super().__init__()
        self.linear = nn.Linear(proj_dim, proj_dim)
        self.proj_dim = proj_dim

    def forward(self, p: torch.Tensor, q: torch.Tensor, q_mask: torch.Tensor):
        """
        Input:
            p: batch_size * p_seq_len * dim
            q: batch_size * q_seq_len * dim
            q_mask: batch_size * q_seq_len (1 for padding, 0 for true)
        Output:
            matched_seq: batch_size * doc_seq_len * dim
        """
        p_transform = F.relu(self.linear(p))
        q_transform = F.relu(self.linear(q))

        # Compute scores s_ij: bsz * doc_seq_len * ques_seq_len
        attn_scores = p_transform.bmm(q_transform.transpose(2, 1))

        # Mask padding: set a very low score for ques tokens that are pads.
        q_mask = q_mask.unsqueeze(1).expand(attn_scores.size())
        attn_scores.data.masked_fill_(q_mask.data, -float("inf"))

        # Normalize with softmax: bsz * doc_seq_len * ques_seq_len
        attn_scores_flattened = F.softmax(attn_scores.view(-1, q.size(1)), dim=-1)
        return attn_scores_flattened.view(-1, p.size(1), q.size(1))


class MultiplicativeAttention(Module):
    """
    Given sequence P and vector q, computes attention weights for each element
    in P by matching q with each element in P using multiplicative attention.
    * a_i = softmax(p_i . W . q)
    """

    class Config(Module.Config):
        p_hidden_dim: int = 32
        q_hidden_dim: int = 32
        normalize: bool = False

    @classmethod
    def from_config(cls, config: Config):
        return cls(config.p_hidden_dim, config.q_hidden_dim, config.normalize)

    def __init__(self, p_hidden_dim, q_hidden_dim, normalize):
        super().__init__()
        self.normalize = normalize
        self.linear = nn.Linear(p_hidden_dim, q_hidden_dim)

    def forward(self, p_seq: torch.Tensor, q: torch.Tensor, p_mask: torch.Tensor):
        """
        Input:
            p_seq: batch_size * p_seq_len * p_hidden_dim
            q: batch_size * q_hidden_dim
            p_mask: batch_size * p_seq_len (1 for padding, 0 for true)
        Output:
            attn_scores: batch_size * p_seq_len
        """
        Wq = self.linear(q) if self.linear is not None else q
        pWq = p_seq.bmm(Wq.unsqueeze(2)).squeeze(2)
        pWq.data.masked_fill_(p_mask.data, -float("inf"))
        attn_scores = F.softmax(pWq, dim=-1) if self.normalize else pWq.exp()
        return attn_scores

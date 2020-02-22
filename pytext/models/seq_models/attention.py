#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
import torch
import torch.nn.functional as F
from pytext.utils.usage import log_class_usage
from torch import nn


def create_src_lengths_mask(batch_size: int, src_lengths):
    """
    Generate boolean mask to prevent attention beyond the end of source

    Inputs:
      batch_size : int
      src_lengths : [batch_size] of sentence lengths

    Outputs:
      [batch_size, max_src_len]
    """
    max_srclen = src_lengths.max()
    src_indices = torch.arange(0, max_srclen).unsqueeze(0).type_as(src_lengths)
    src_indices = src_indices.expand(batch_size, max_srclen)
    src_lengths = src_lengths.unsqueeze(dim=1).expand(batch_size, max_srclen)
    # returns [batch_size, max_seq_len]
    return (src_indices < src_lengths).int().detach()


def masked_softmax(scores, src_lengths, src_length_masking: bool = True):
    """Apply source length masking then softmax.
    Input and output have shape bsz x src_len"""
    if src_length_masking:
        bsz, max_src_len = scores.size()
        # compute masks
        src_mask = create_src_lengths_mask(bsz, src_lengths)
        # Fill pad positions with -inf
        scores = scores.masked_fill(src_mask == 0, -np.inf)

    # Cast to float and then back again to prevent loss explosion under fp16.
    return F.softmax(scores.float(), dim=-1).type_as(scores)


class DotAttention(nn.Module):
    def __init__(
        self,
        decoder_hidden_state_dim,
        context_dim,
        force_projection=False,
        src_length_masking=True,
    ):
        super().__init__()
        self.decoder_hidden_state_dim = decoder_hidden_state_dim
        self.context_dim = context_dim

        self.input_proj = None
        if force_projection or decoder_hidden_state_dim != context_dim:
            self.input_proj = nn.Linear(
                decoder_hidden_state_dim, context_dim, bias=True
            )
        self.src_length_masking = src_length_masking
        log_class_usage(__class__)

    def forward(self, decoder_state, source_hids, src_lengths):
        # Reshape to bsz x src_len x context_dim
        source_hids = source_hids.transpose(0, 1)
        # decoder_state: bsz x context_dim
        if self.input_proj is not None:
            decoder_state = self.input_proj(decoder_state)
        # compute attention (bsz x src_len x context_dim) * (bsz x context_dim x 1)
        attn_scores = torch.bmm(source_hids, decoder_state.unsqueeze(2)).squeeze(2)

        # Mask + softmax (bsz x src_len)
        normalized_masked_attn_scores = masked_softmax(
            attn_scores, src_lengths, self.src_length_masking
        )

        # Sum weighted sources
        attn_weighted_context = (
            (source_hids * normalized_masked_attn_scores.unsqueeze(2))
            .contiguous()
            .sum(1)
        )

        return attn_weighted_context, normalized_masked_attn_scores.t()

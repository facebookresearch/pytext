#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from pytext.config import ConfigBase
from torch import Tensor, nn

from .base import PyTextIncrementalDecoderComponent, PyTextSeq2SeqModule
from .utils import Linear


class EncoderDecoderMultiheadAttention(
    PyTextSeq2SeqModule, PyTextIncrementalDecoderComponent
):
    """
    Refer Attention is All You Need for more details.

    This is a simplified implementation of encoder-decoder attention
    optimized for exporting using torchscript. Usage of nn.Linear() instead of
    F.Linear() helps to quantize the linear layers.

    Query represents the output from last decoder step. Key and Values are obtained from
    encoder. Attention weights are obtained from the dot product of query and key.
    Attention weights multiplied by the value gives output.
    """

    class Config(ConfigBase):
        dropout: float = 0.0
        kdim: Optional[int] = None
        vdim: Optional[int] = None
        bias: bool = True

    @classmethod
    def from_config(cls, config, embed_dim, num_heads):
        return cls(embed_dim, num_heads, **config._asdict())

    def __init__(self, embed_dim, num_heads, dropout, kdim=None, vdim=None, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = embed_dim if kdim is None else kdim
        self.vdim = embed_dim if vdim is None else vdim

        self.q_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = Linear(self.vdim, embed_dim, bias=bias)

        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        query,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor],
        need_weights: bool,
        incremental_state: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        target_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        assert embed_dim == self.embed_dim, (
            str(embed_dim) + " != " + str(self.embed_dim)
        )
        assert key is not None
        assert value is not None

        if incremental_state is not None:
            prev_key = self._get_input_buffer(incremental_state, "prev_key")
        else:
            prev_key = None

        bsz_X_num_heads = bsz * self.num_heads

        q = self.q_proj(query)
        q *= self.scaling
        q = (
            q.contiguous()
            .view(target_len, bsz_X_num_heads, self.head_dim)
            .transpose(0, 1)
        )

        if prev_key is not None and incremental_state is not None:
            # This happens if its incremental decoding and prev time step has been
            # cached. This condition won't be true for the first step in
            # incremental decoding.
            k = prev_key.view(bsz_X_num_heads, -1, self.head_dim)

            prev_value = self._get_input_buffer(incremental_state, "prev_value")
            assert prev_value is not None

            v = prev_value.view(bsz_X_num_heads, -1, self.head_dim)

        else:
            # We will recompute key and value for all regular training and
            # for first step of incremental decoding
            k = self.k_proj(key)

            k = k.contiguous().view(-1, bsz_X_num_heads, self.head_dim).transpose(0, 1)

            v = self.v_proj(value)
            v = v.contiguous().view(-1, bsz_X_num_heads, self.head_dim).transpose(0, 1)

            # incremental state needs to be set only for the first decoder step
            # when prev_key and prev_value was not present in incremental_state
            if incremental_state is not None:
                self._set_input_buffer(
                    incremental_state,
                    "prev_key",
                    k.view(bsz, self.num_heads, -1, self.head_dim),
                )
                self._set_input_buffer(
                    incremental_state,
                    "prev_value",
                    v.view(bsz, self.num_heads, -1, self.head_dim),
                )
                key_padding_mask = self._get_input_buffer(
                    incremental_state, "prev_key_padding_mask"
                )
                if key_padding_mask is not None:
                    self._set_input_buffer(
                        incremental_state, "prev_key_padding_mask", key_padding_mask
                    )

        # q.size() : bsz_X_num_heads, target_len, self.head_dim
        assert list(k.size()) == [bsz_X_num_heads, src_len, self.head_dim], (
            f"key.size() :{ k.size()} "
            f"[ bsz_X_num_heads, src_len, self.head_dim] : "
            f"[{bsz_X_num_heads}, {src_len}, {self.head_dim}]"
        )

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        # attn_weights.size() : bsz_X_num_heads, target_len, src_len

        # Don't attend to padding symbols
        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, target_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )
            attn_weights = attn_weights.view(bsz_X_num_heads, target_len, src_len)

        assert list(attn_weights.size()) == [bsz_X_num_heads, target_len, src_len]

        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_probs = self.dropout(attn_weights)

        attn = torch.bmm(attn_probs, v)

        assert list(attn.size()) == [bsz_X_num_heads, target_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(target_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        if need_weights:
            attn_weights = attn_weights.view(
                bsz, self.num_heads, target_len, src_len
            ).transpose(1, 0)
            attn_weights_out = attn_weights.mean(dim=0)
        else:
            attn_weights_out = None

        return attn, attn_weights_out

    def reorder_incremental_state(
        self, incremental_state: Dict[str, Tensor], new_order: Tensor
    ):
        """Reorder buffered internal state (for incremental generation)."""
        all_keys = ["prev_key", "prev_value", "prev_key_padding_mask"]
        # ARBABU : why do we need to reorder_incremental_state as encoder_out
        # is always the same?
        for key in all_keys:
            input_buffer = self._get_input_buffer(incremental_state, key)
            if input_buffer is not None:
                # During incremental decoding, all candidates will be along
                # the batch dimension. We pick top candidates
                input_buffer = input_buffer.index_select(0, new_order)
                self._set_input_buffer(incremental_state, key, input_buffer)

    def _get_input_buffer(self, incremental_state: Dict[str, Tensor], key: str):
        return self.get_incremental_state(incremental_state, key)

    def _set_input_buffer(
        self, incremental_state: Dict[str, Tensor], key: str, value: Tensor
    ):
        self.set_incremental_state(incremental_state, key, value)


class DecoupledMultiheadAttention(nn.Module):
    """
    Multiheaded Scaled Dot Product Attention. This function
    has the same exact signature as the one used in pytorch_translate
    with the added benefit of supporting torchscript.
    """

    def __init__(
        self,
        embed_dim: int,
        context_dim: int,
        num_heads: int,
        dropout: float,
        unseen_mask=False,
        src_length_mask=True,
    ):
        super().__init__()
        assert embed_dim == context_dim
        d_model = embed_dim
        assert d_model % num_heads == 0

        if unseen_mask:
            raise NotImplementedError(
                "Unseen mask not supported with sequential decoding"
            )
        self._attn = EncoderDecoderMultiheadAttention(d_model, num_heads, dropout)
        self.use_src_length_mask = src_length_mask

    def forward(
        self,
        decoder_state: Tensor,
        source_hids: Tensor,
        src_len_mask: Optional[Tensor],
        squeeze: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        """
        Computes MultiheadAttention with respect to either a vector
        or a tensor

        Inputs:
            decoder_state: (bsz x decoder_hidden_state_dim) or
            (bsz x T x decoder_hidden_state_dim)

            source_hids: srclen x bsz x context_dim

            src_lengths: bsz x 1, actual sequence lengths

            squeeze: Whether or not to squeeze on the time dimension.
            Even if decoder_state.dim() is 2 dimensional an
            explicit time step dimension will be unsqueezed.

        Outputs:
          [batch_size, max_src_len] if decoder_state.dim() == 2 & squeeze
          or [batch_size, 1, max_src_len] if decoder_state.dim() == 2 & !squeeze
          or [batch_size, T, max_src_len] if decoder_state.dim() == 3 & !squeeze
          or [batch_size, T, max_src_len] if decoder_state.dim() == 3 & squeeze & T != 1
          or [batch_size, max_src_len] if decoder_state.dim() == 3 & squeeze & T == 1

        """
        if decoder_state.dim() == 3:
            query = decoder_state
        elif decoder_state.dim() == 2:
            query = decoder_state.unsqueeze(1)
        else:
            raise ValueError("decoder state must be either 2 or 3 dimensional")
        query = query.transpose(0, 1)
        value = key = source_hids

        attn, attn_weights = self._attn.forward(
            query, key, value, key_padding_mask=src_len_mask, need_weights=True
        )
        # Need to satify torchscript here
        if attn_weights is None:
            raise NotImplementedError("")
        # attn.shape = T X bsz X embed_dim
        # attn_weights.shape = bsz X T X src_len

        attn_weights = attn_weights.transpose(0, 2)
        # attn_weights.shape = src_len X T X bsz

        if squeeze:
            attn = attn.squeeze(0)
            # attn.shape = squeeze(T) X bsz X embed_dim
            attn_weights = attn_weights.squeeze(1)
            # attn_weights.shape = src_len X squeeze(T) X bsz
            return attn, attn_weights
        return attn, attn_weights


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
    def __init__(self, decoder_hidden_state_dim, context_dim, **kwargs):
        super().__init__()
        self.decoder_hidden_state_dim = decoder_hidden_state_dim
        self.context_dim = context_dim

        self.input_proj = None
        force_projection = kwargs.get("force_projection", False)
        if force_projection or decoder_hidden_state_dim != context_dim:
            self.input_proj = Linear(decoder_hidden_state_dim, context_dim, bias=True)
        self.src_length_masking = kwargs.get("src_length_masking", True)

    def prepare_for_onnx_export_(self, **kwargs):
        self.src_length_masking = False

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

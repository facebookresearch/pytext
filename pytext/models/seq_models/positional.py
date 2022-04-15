#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
from enum import Enum
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytext.models.representations.transformer import (
    PositionalEmbedding,
)
from pytext.models.seq_models.base import PlaceholderIdentity
from pytext.models.seq_models.base import (
    PlaceholderIdentity,
)
from torch import Tensor

from .utils import make_positions


class PostionalEmbedType(Enum):
    LEARNED = "learned"
    SINUSOIDAL = "sinusoidal"
    HYBRID = "hybrid"


class PostionalEmbedCombine(Enum):
    SUM = "sum"
    CONCAT = "concat"


def get_sinusoidal_embedding(num_embeddings: int, embedding_dim: int, padding_idx: int):
    """Build sinusoidal embeddings.

    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
    emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(
        0
    )
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
    if embedding_dim % 2 == 1:
        # zero pad
        emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)

    # Positions < padding_idx are ignored and won't be used.
    emb[padding_idx, :] = 0
    return emb


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=124, learned_embed=False):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        if not learned_embed:
            self.sinusoidal_embedding_dim = embedding_dim
            self.learned_embed = learned_embed
            self.learned_embedding = PlaceholderIdentity()
        else:
            assert embedding_dim % 2 == 0
            self.sinusoidal_embedding_dim = embedding_dim // 2
            self.learned_embedding = nn.Embedding(
                init_size, embedding_dim // 2, padding_idx
            )
            self.learned_embed = learned_embed
        self.weights = nn.Parameter(
            get_sinusoidal_embedding(
                init_size, self.sinusoidal_embedding_dim, padding_idx
            )
        )
        self.weights.requires_grad = False
        self.max_positions = int(1e5)  # an arbitrary large number

    def forward(
        self,
        input,
        incremental_state: Optional[Dict[str, Tensor]] = None,
        timestep: Optional[int] = None,
    ):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input.size()
        max_pos = self.padding_idx + 1 + seq_len
        assert max_pos < self.weights.size(
            0
        ), f"max_pos :{max_pos}, self.weights.size(0): {self.weights.size(0)}"

        if incremental_state is not None:
            # Positions is the same for every token when decoding a single step
            # Either its timestep + 1 or len(prev_tokens)
            pos = timestep if timestep is not None else seq_len
            assert pos != 0, "Position cannot start from 0"
            return self.weights[self.padding_idx + pos, :].expand(bsz, 1, -1)

        positions = make_positions(input, self.padding_idx)
        sinusoidal_embedding = (
            self.weights.index_select(0, positions.view(-1))
            .view(bsz, seq_len, -1)
            .detach()
        )
        if self.learned_embed:
            learned_embedding = self.learned_embedding(positions)
            embed_out = torch.cat([sinusoidal_embedding, learned_embedding], dim=-1)
        else:
            embed_out = sinusoidal_embedding
        return embed_out


class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        if self.padding_idx is not None:
            self.max_positions = self.num_embeddings - self.padding_idx - 1
        else:
            self.max_positions = self.num_embeddings

    def forward(
        self,
        input: Tensor,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        positions: Optional[Tensor] = None,
    ):
        """Input is expected to be of size [bsz x seqlen]."""
        assert (positions is None) or (
            self.padding_idx is None
        ), "If positions is pre-computed then padding_idx should not be set."

        if positions is None:
            if incremental_state is not None:
                # positions is the same for every token when decoding a single step
                # Without the int() cast, it doesn't work in some cases when exporting to ONNX
                positions = torch.zeros(
                    (1, 1), device=input.device, dtype=input.dtype
                ).fill_(int(self.padding_idx + input.size(1)))
            else:
                positions = make_positions(input, self.padding_idx)
        return F.embedding(
            positions,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


def build_positional_embedding(
    positional_embedding_type: PostionalEmbedType,
    combine_pos_embed: PostionalEmbedCombine,
    max_target_positions: int,
    input_embed_dim: int,
    embed_dim: int,
    padding_idx: int,
    no_token_positional_embeddings: bool,
):
    if combine_pos_embed == PostionalEmbedCombine.SUM:
        pos_embed_dim = embed_dim
    elif combine_pos_embed == PostionalEmbedCombine.CONCAT:
        pos_embed_dim = embed_dim - input_embed_dim
    else:
        raise NotImplementedError
    if not no_token_positional_embeddings:
        if positional_embedding_type == PostionalEmbedType.LEARNED:
            return PositionalEmbedding(
                max_target_positions,
                pos_embed_dim,
                padding_idx,
            )
        elif (
            positional_embedding_type == PostionalEmbedType.SINUSOIDAL
            or positional_embedding_type == PostionalEmbedType.HYBRID
        ):
            return SinusoidalPositionalEmbedding(
                pos_embed_dim,
                padding_idx,
                init_size=max_target_positions,
                learned_embed=positional_embedding_type == PostionalEmbedType.HYBRID,
            )
        else:
            raise NotImplementedError("Positional embedding type not supported")
    else:
        return PlaceholderIdentity()

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import math
from typing import Optional, Dict, List

import torch
import torch.nn as nn
from fairseq.modules import (
    PositionalEmbedding,
    LayerNorm,
)
from pytext.models.module import create_module
from pytext.models.seq_models.base import PlaceholderIdentity
from pytext.models.seq_models.conv_encoder import (
    LightConvEncoderLayer,
    ConvEncoderConfig,
)
from pytext.models.seq_models.positional import (
    PostionalEmbedCombine,
    PostionalEmbedType,
    SinusoidalPositionalEmbedding,
)
from pytext.utils.usage import log_class_usage
from torch import Tensor
from torch.nn import Linear

from .representation_base import RepresentationBase


def mean(rep: Tensor, padding_mask: Optional[Tensor]):
    rep_sum = rep.sum(dim=1)  # B x T x C => B x C
    if padding_mask is not None:
        lengths = (~padding_mask).sum(dim=1).reshape(-1, 1)
    else:
        bsz, max_token_len, _embed_dim = rep.size()
        lengths = torch.full(
            (bsz, 1), max_token_len, dtype=torch.long, device=rep.device
        )

    return rep_sum / lengths


def pool(pooling_type: str, words: Tensor, encoder_padding_mask: Optional[Tensor]):
    # input dims: bsz * seq_len * num_filters
    if pooling_type == "mean":
        return mean(words, encoder_padding_mask)
    elif pooling_type == "max":
        return words.max(dim=1)[0]
    elif pooling_type == "none":
        return words
    else:
        raise NotImplementedError


class LightConvRepresentation(RepresentationBase):
    """CNN based representation of a document."""

    class Config(RepresentationBase.Config):
        encoder_config: ConvEncoderConfig = ConvEncoderConfig()
        layer_config: LightConvEncoderLayer.Config = LightConvEncoderLayer.Config()
        encoder_kernel_size_list: List[int] = [3, 7, 15]
        pooling_type: str = "mean"

    def __init__(self, config: Config, embed_dim: int, padding_idx: Tensor) -> None:
        super().__init__(config)
        self.padding_idx = padding_idx
        self.pooling_type = config.pooling_type
        self.dropout = nn.Dropout(config.encoder_config.dropout)
        input_embed_dim = embed_dim
        self.embed_scale = math.sqrt(input_embed_dim)  # todo: try with input_embed_dim
        self.max_source_positions = config.encoder_config.max_source_positions
        self.no_token_positional_embeddings = (
            config.encoder_config.no_token_positional_embeddings
        )

        # creating this is also conditional
        self.project_in_dim = (
            Linear(input_embed_dim, config.encoder_config.encoder_embed_dim)
            if config.encoder_config.encoder_embed_dim != input_embed_dim
            else PlaceholderIdentity()
        )

        layers = []
        # Overwrite the config.layer_config.encoder_embed_dim so that it will always match with config.encoder_config.encoder_embed_dim
        config.layer_config.encoder_embed_dim = config.encoder_config.encoder_embed_dim
        for size in config.encoder_kernel_size_list:
            layers.append(create_module(config.layer_config, kernel_size=size))

        self.layers = nn.ModuleList(layers)
        self.embed_layer_norm = LayerNorm(config.encoder_config.encoder_embed_dim)
        self.combine_pos_embed = config.encoder_config.combine_pos_embed.value

        if config.encoder_config.combine_pos_embed == PostionalEmbedCombine.SUM:
            pos_embed_dim = config.encoder_config.encoder_embed_dim
        elif config.encoder_config.combine_pos_embed == PostionalEmbedCombine.CONCAT:
            pos_embed_dim = config.encoder_config.encoder_embed_dim - input_embed_dim
        else:
            raise NotImplementedError

        if not config.encoder_config.no_token_positional_embeddings:
            if (
                config.encoder_config.positional_embedding_type
                == PostionalEmbedType.LEARNED
            ):
                self.embed_positions = PositionalEmbedding(
                    config.encoder_config.max_source_positions,
                    pos_embed_dim,
                    self.padding_idx,
                )
            elif (
                config.encoder_config.positional_embedding_type
                == PostionalEmbedType.SINUSOIDAL
                or config.encoder_config.positional_embedding_type
                == PostionalEmbedType.HYBRID
            ):
                self.embed_positions = SinusoidalPositionalEmbedding(
                    pos_embed_dim,
                    self.padding_idx,
                    init_size=config.encoder_config.max_source_positions,
                    learned_embed=config.encoder_config.positional_embedding_type
                    == PostionalEmbedType.HYBRID,
                )
            else:
                raise NotImplementedError("Positional embedding type not supported")
        else:
            self.embed_positions = PlaceholderIdentity()

        self.normalize = config.encoder_config.encoder_normalize_before
        if self.normalize:
            self.layer_norm = LayerNorm(config.encoder_config.encoder_embed_dim)
        else:
            self.layer_norm = PlaceholderIdentity()

        log_class_usage(__class__)

    def forward(
        self, embedded_tokens: torch.Tensor, src_tokens: Tensor, src_lengths: Tensor
    ) -> torch.Tensor:

        x = self.embed_scale * embedded_tokens
        if not self.no_token_positional_embeddings:
            x = self.pos_embed(x, src_tokens)
        else:
            x = self.project_in_dim(x)

        x = self.embed_layer_norm(x)
        x = self.dropout(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # Compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)  # B x T
        if (
            not encoder_padding_mask.any()
        ):  # Setting to None helps us avoid some masking operations later.
            # Different name is used to avoid some torchscript type checks
            encoder_mask = None
        else:
            encoder_mask = encoder_padding_mask

        # Encoder layers
        for _, layer in enumerate(self.layers):
            x = layer(x, encoder_mask)

        if self.normalize:
            x = self.layer_norm(x)

        x = pool(self.pooling_type, x.transpose(0, 1), encoder_padding_mask)
        return x

    def reorder_encoder_out(self, encoder_out: Dict[str, Tensor], new_order: Tensor):
        encoder = encoder_out["encoder_out"]
        encoder = encoder.index_select(1, new_order)
        output_dict = {"encoder_out": encoder}

        output_dict["src_tokens"] = encoder_out["src_tokens"].index_select(0, new_order)
        padding_mask = encoder_out.get("encoder_mask", None)
        if padding_mask is not None:
            padding_mask = padding_mask.index_select(0, new_order)
            output_dict["encoder_mask"] = padding_mask
        return output_dict

    def pos_embed(self, x, src_tokens):
        if self.combine_pos_embed == PostionalEmbedCombine.SUM.value:
            x = self.project_in_dim(x)
            return self._vanilla_transformer(x, src_tokens)
        elif self.combine_pos_embed == PostionalEmbedCombine.CONCAT.value:
            return self._concat_pos_embed(x, src_tokens)
        else:
            raise NotImplementedError("Method not supported")

    def _vanilla_transformer(self, x, src_tokens):
        x += self.embed_positions(src_tokens)
        return x

    def _concat_pos_embed(self, x, src_tokens):
        pos_embed = self.embed_positions(src_tokens)
        return torch.cat([x, pos_embed], dim=2)

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.no_token_positional_embeddings:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions)

    def tile_encoder_out(
        self, tile_size: int, encoder_out: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        tiled_out = torch.jit.annotate(Dict[str, Tensor], {})

        x = encoder_out["encoder_out"]
        new_x = x.repeat(1, tile_size, 1)
        tiled_out["encoder_out"] = new_x

        if "encoder_mask" in encoder_out:
            new_encoder_mask = encoder_out["encoder_mask"].repeat(tile_size, 1)
            tiled_out["encoder_mask"] = new_encoder_mask
        if "src_tokens" in encoder_out:
            tiled_out["src_tokens"] = encoder_out["src_tokens"].repeat(tile_size, 1)
        if "src_lengths" in encoder_out:
            tiled_out["src_lengths"] = encoder_out["src_lengths"].repeat(tile_size, 1)

        return tiled_out

    def extra_repr(self):
        s = "dropout={}, embed_scale={}, normalize={}".format(
            self.dropout, self.embed_scale, self.normalize
        )
        return s

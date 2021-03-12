#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytext.config import ConfigBase
from pytext.config.module_config import ModuleConfig
from pytext.models.module import create_module
from pytext.models.representations.transformer.positional_embedding import (
    PositionalEmbedding,
)
from pytext.models.seq_models.base import (
    PlaceholderAttentionIdentity,
    PlaceholderIdentity,
)
from pytext.models.seq_models.positional import (
    PostionalEmbedCombine,
    PostionalEmbedType,
    SinusoidalPositionalEmbedding,
)
from torch import Tensor
from torch.nn import LayerNorm

from .attention import MultiheadAttention
from .base import PyTextSeq2SeqModule, PlaceholderIdentity
from .light_conv import LightweightConv
from .nar_modules import NAREncoderUtility
from .utils import Linear


class LightConvEncoderLayer(PyTextSeq2SeqModule):
    class Config(ConfigBase):
        dropout: float = 0.1
        encoder_conv_dim: int = 128
        encoder_conv_type: Union[
            LightweightConv.Config, PlaceholderIdentity.Config
        ] = LightweightConv.Config()
        self_attention_type: Optional[MultiheadAttention.Config] = None
        encoder_embed_dim: int = 128
        encoder_ffn_embed_dim: int = 512
        self_attention_heads: int = 2
        encoder_glu: bool = True
        encoder_normalize_before: bool = False
        input_dropout: float = 0.1
        relu_dropout: float = 0.0
        convolution_type: str = "non-causal"

    @classmethod
    def from_config(cls, config, kernel_size):
        conv = create_module(
            config.encoder_conv_type,
            input_size=config.encoder_conv_dim,
            kernel_size=kernel_size,
            convolution_type="non-causal",
        )
        if config.self_attention_type is not None:
            self_attention = create_module(
                config.self_attention_type,
                config.encoder_embed_dim,
                config.self_attention_heads,
            )
        else:
            self_attention = None
        return cls(**config._asdict(), conv=conv, self_attention=self_attention)

    def __init__(
        self,
        dropout,
        encoder_conv_dim,
        encoder_conv_type,
        self_attention_type,
        encoder_embed_dim,
        encoder_ffn_embed_dim,
        self_attention_heads,
        encoder_glu,
        encoder_normalize_before,
        input_dropout,
        relu_dropout,
        convolution_type,
        conv=None,
        self_attention=None,
    ):
        super().__init__()
        self.embed_dim = encoder_embed_dim
        self.conv_dim = encoder_conv_dim

        if encoder_glu:
            self.linear1 = Linear(self.embed_dim, 2 * self.conv_dim)
            self.act = nn.GLU()
        else:
            self.linear1 = Linear(self.embed_dim, self.conv_dim)
            self.act = None
        self.conv = conv
        self.linear2 = Linear(self.conv_dim, self.embed_dim)
        self.dropout = dropout
        self.relu_dropout = relu_dropout
        self.input_dropout = input_dropout
        self.normalize_before = encoder_normalize_before
        self.fc1 = Linear(self.embed_dim, encoder_ffn_embed_dim)
        self.fc2 = Linear(encoder_ffn_embed_dim, self.embed_dim)

        self.layer_norm1 = LayerNorm(self.embed_dim)
        self.layer_norm2 = LayerNorm(self.embed_dim)
        if self_attention is None:
            self.has_self_attn = False
            self.self_attn = PlaceholderAttentionIdentity()
        else:
            self.has_self_attn = True
            self.self_attn = self_attention

    def forward(self, x, encoder_padding_mask: Optional[Tensor] = None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual = x

        normalize = self.maybe_layer_norm(before=True)
        if normalize:
            x = self.layer_norm1(x)
        if self.has_self_attn:
            x, _ = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask,
                incremental_state=None,
                need_weights=False,
            )
            x = residual + x
            residual = x
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.linear1(x)
        x = self.act(x)
        if encoder_padding_mask is not None:
            x = x.masked_fill(encoder_padding_mask.transpose(0, 1).unsqueeze(2), 0)
        x = self.conv(x)
        x = self.linear2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        normalize = self.maybe_layer_norm(after=True)
        if normalize:
            x = self.layer_norm1(x)

        residual = x
        normalize = self.maybe_layer_norm(before=True)
        if normalize:
            x = self.layer_norm2(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        normalize = self.maybe_layer_norm(after=True)
        if normalize:
            x = self.layer_norm2(x)
        return x

    def maybe_layer_norm(self, before: bool = False, after: bool = False):
        assert before ^ after, "Incorrect arguments"
        return after ^ self.normalize_before

    def extra_repr(self):
        return "dropout={}, relu_dropout={}, input_dropout={}, normalize_before={}".format(  # noqa
            self.dropout, self.relu_dropout, self.input_dropout, self.normalize_before
        )


class ConvEncoderConfig(ConfigBase):
    dropout: float = 0.1
    encoder_learned_pos: bool = False
    encoder_normalize_before: bool = False
    max_source_positions: int = 1024
    max_target_positions: int = 100
    no_token_positional_embeddings: bool = False
    positional_embedding_type: PostionalEmbedType = PostionalEmbedType.LEARNED
    combine_pos_embed: PostionalEmbedCombine = PostionalEmbedCombine.CONCAT
    encoder_embed_dim: Optional[int] = 128
    embedding_dim: Optional[int] = 128


class LightConvEncoder(PyTextSeq2SeqModule, NAREncoderUtility):
    class Config(ModuleConfig):
        encoder_config: ConvEncoderConfig = ConvEncoderConfig()
        layer_config: LightConvEncoderLayer.Config = LightConvEncoderLayer.Config()
        encoder_kernel_size_list: List[int] = [3, 7, 15]
        compression_dim: Optional[int] = 128

    @classmethod
    def from_config(cls, config, src_dict, src_embedding):
        kernel_size_list = config.encoder_kernel_size_list
        layers = []
        for size in kernel_size_list:
            assert (
                config.encoder_config.encoder_embed_dim
                == config.layer_config.encoder_embed_dim
            )
            layers.append(create_module(config.layer_config, kernel_size=size))
        return cls(src_dict, src_embedding, layers, config.encoder_config)

    def __init__(self, dictionary, embed_tokens, layers, encoder_config):
        super().__init__()
        self.dropout = encoder_config.dropout

        input_embed_dim = embed_tokens.embedding_dim
        self.padding_idx = dictionary.get_pad_index()
        self.max_source_positions = encoder_config.max_source_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(input_embed_dim)  # todo: try with input_embed_dim
        self.no_token_positional_embeddings = (
            encoder_config.no_token_positional_embeddings
        )
        # creating this is also conditional
        self.project_in_dim = (
            Linear(input_embed_dim, encoder_config.encoder_embed_dim)
            if encoder_config.encoder_embed_dim != input_embed_dim
            else PlaceholderIdentity()
        )
        self.embed_layer_norm = LayerNorm(encoder_config.encoder_embed_dim)

        self.combine_pos_embed = encoder_config.combine_pos_embed.value
        if encoder_config.combine_pos_embed == PostionalEmbedCombine.SUM:
            pos_embed_dim = encoder_config.encoder_embed_dim
        elif encoder_config.combine_pos_embed == PostionalEmbedCombine.CONCAT:
            pos_embed_dim = encoder_config.encoder_embed_dim - input_embed_dim
        else:
            raise NotImplementedError

        if not encoder_config.no_token_positional_embeddings:
            if encoder_config.positional_embedding_type == PostionalEmbedType.LEARNED:
                self.embed_positions = PositionalEmbedding(
                    encoder_config.max_source_positions,
                    pos_embed_dim,
                    self.padding_idx,
                )
            elif (
                encoder_config.positional_embedding_type
                == PostionalEmbedType.SINUSOIDAL
                or encoder_config.positional_embedding_type == PostionalEmbedType.HYBRID
            ):
                self.embed_positions = SinusoidalPositionalEmbedding(
                    pos_embed_dim,
                    self.padding_idx,
                    init_size=encoder_config.max_source_positions,
                    learned_embed=encoder_config.positional_embedding_type
                    == PostionalEmbedType.HYBRID,
                )
            else:
                raise NotImplementedError("Positional embedding type not supported")
        else:
            self.embed_positions = PlaceholderIdentity()

        self.layers = nn.ModuleList(layers)

        self.normalize = encoder_config.encoder_normalize_before
        if self.normalize:
            self.layer_norm = LayerNorm(input_embed_dim)
        else:
            self.layer_norm = PlaceholderIdentity()

    def forward(
        self, src_tokens: Tensor, src_embeddings: Tensor, src_lengths: Tensor
    ) -> Dict[str, Tensor]:
        output_dict: Dict[str, Tensor] = {}

        # embed tokens and positions
        x = self.embed_scale * src_embeddings
        if not self.no_token_positional_embeddings:
            x = self.pos_embed(x, src_tokens)
        else:
            x = self.project_in_dim(x)

        x = self.embed_layer_norm(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        output_dict["encoder_layer_0"] = x.clone()

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
        for idx, layer in enumerate(self.layers):
            x = layer(x, encoder_mask)
            output_dict["encoder_layer_" + str(idx + 1)] = x.transpose(0, 1).clone()

        if self.normalize:
            x = self.layer_norm(x)

        output_dict["src_tokens"] = src_tokens  # B x T
        if src_lengths is not None:
            output_dict["src_lengths"] = src_lengths
        output_dict["encoder_out"] = x  # T x B x C
        if encoder_mask is not None:
            output_dict["encoder_mask"] = encoder_mask  # B x T
        return output_dict

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
        return min(self.max_source_positions, self.embed_positions.max_positions())

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

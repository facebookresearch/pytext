#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytext.config import ConfigBase
from pytext.config.module_config import ModuleConfig
from pytext.models.module import create_module
from pytext.models.seq_models.base import (
    PlaceholderAttentionIdentity,
)
from pytext.models.seq_models.positional import (
    PostionalEmbedCombine,
    PostionalEmbedType,
    build_positional_embedding,
)
from pytext.models.seq_models.utils import Linear, log_and_overwrite
from torch import Tensor
from torch.nn import LayerNorm

from .attention import MultiheadAttention
from .base import (
    PyTextIncrementalDecoderComponent,
    PyTextSeq2SeqModule,
    PlaceholderIdentity,
)
from .light_conv import LightweightConv
from .projection_layers import (
    DecoderWithLinearOutputProjection,
    DecoupledDecoderHead,
)
from .utils import extract_ontology_vocab


class LightConvDecoderLayer(PyTextSeq2SeqModule):
    class Config(ConfigBase):
        attention_dropout: float = 0.0
        decoder_attention_heads: int = 1
        self_attention_heads: int = 1
        decoder_conv_dim: int = 128
        decoder_conv_type: Union[
            LightweightConv.Config, PlaceholderIdentity.Config
        ] = LightweightConv.Config()
        attention_type: Union[
            MultiheadAttention.Config, None
        ] = MultiheadAttention.Config()
        self_attention_type: Optional[MultiheadAttention.Config] = None
        decoder_embed_dim: int = 128
        decoder_ffn_embed_dim: int = 512
        decoder_glu: bool = True
        decoder_normalize_before: bool = False
        dropout: float = 0.1
        input_dropout: float = 0.1
        relu_dropout: float = 0.0
        need_attention: bool = True
        convolution_type: str = "causal"

    @classmethod
    def from_config(cls, config, kernel_size):
        conv = create_module(
            config.decoder_conv_type,
            input_size=config.decoder_conv_dim,
            kernel_size=kernel_size,
            convolution_type=config.convolution_type,
        )
        if config.attention_type is not None:
            attention = create_module(
                config.attention_type,
                config.decoder_embed_dim,
                config.decoder_attention_heads,
            )
        else:
            attention = None
        if config.self_attention_type is not None:
            self_attention = create_module(
                config.self_attention_type,
                config.decoder_embed_dim,
                config.self_attention_heads,
            )
        else:
            self_attention = None
        return cls(
            **config._asdict(),
            conv=conv,
            self_attention=self_attention,
            attention=attention
        )

    def __init__(
        self,
        attention_dropout,
        decoder_attention_heads,
        self_attention_heads,
        decoder_conv_dim,
        # ARBABU: need to remove these two type parameters
        decoder_conv_type,
        attention_type,
        self_attention_type,
        decoder_embed_dim,
        decoder_ffn_embed_dim,
        decoder_glu,
        decoder_normalize_before,
        dropout,
        input_dropout,
        relu_dropout,
        need_attention,
        convolution_type,
        conv=None,
        self_attention=None,
        attention=None,
    ):
        super().__init__()
        self.embed_dim = decoder_embed_dim
        self.conv_dim = decoder_conv_dim
        if decoder_glu:
            self.linear1 = Linear(self.embed_dim, 2 * self.conv_dim)
            self.act = nn.GLU()
        else:
            self.linear1 = Linear(self.embed_dim, self.conv_dim)
            self.act = PlaceholderIdentity()
        self.conv = conv
        self.linear2 = Linear(self.conv_dim, self.embed_dim)

        self.dropout = dropout
        self.relu_dropout = relu_dropout
        self.input_dropout = input_dropout
        self.normalize_before = decoder_normalize_before
        self.conv_layer_norm = LayerNorm(self.embed_dim)

        if attention is None:
            self.no_encoder_attn = True
            self.encoder_attn = PlaceholderAttentionIdentity()
            self.encoder_attn_layer_norm = PlaceholderIdentity()
        else:
            self.no_encoder_attn = False
            self.encoder_attn = attention
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)
        if self_attention is None:
            self.has_self_attn = False
            self.self_attn = PlaceholderAttentionIdentity()
        else:
            self.has_self_attn = True
            self.self_attn = self_attention
        self.fc1 = Linear(self.embed_dim, decoder_ffn_embed_dim)
        self.fc2 = Linear(decoder_ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = LayerNorm(self.embed_dim)
        self.need_attn = need_attention

    def forward(
        self,
        x,
        encoder_out: Tensor,
        encoder_padding_mask: Optional[Tensor],
        decoder_padding_mask: Optional[Tensor],
        incremental_state: Optional[Dict[str, Tensor]],
    ):
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
            x = self.conv_layer_norm(x)
        if self.has_self_attn:
            x, _ = self.self_attn(
                x,
                key=x,
                value=x,
                key_padding_mask=decoder_padding_mask,
                need_weights=False,
                incremental_state=incremental_state,
            )
            x = residual + x
            residual = x
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.linear1(x)
        x = self.act(x)
        if decoder_padding_mask is not None:
            x = x.masked_fill(decoder_padding_mask.transpose(0, 1).unsqueeze(2), 0)
        x = self.conv(x, incremental_state=incremental_state)
        x = self.linear2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        normalize = self.maybe_layer_norm(after=True)
        if normalize:
            x = self.conv_layer_norm(x)

        attn: Optional[Tensor] = None
        if not self.no_encoder_attn:
            residual = x
            normalize = self.maybe_layer_norm(before=True)
            if normalize:
                x = self.encoder_attn_layer_norm(x)
            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                need_weights=(not self.training and self.need_attn),
            )
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            normalize = self.maybe_layer_norm(after=True)
            if normalize:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        normalize = self.maybe_layer_norm(before=True)
        if normalize:
            x = self.final_layer_norm(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        normalize = self.maybe_layer_norm(after=True)
        if normalize:
            x = self.final_layer_norm(x)
        return x, attn

    def maybe_layer_norm(self, before: bool = False, after: bool = False):
        """This a utility function which helps to control the layer norm behavior
        `before` and `after` specific components using one variable in config.
        If self.normalize_before is set to True, output is true only when `before`
        is True
        """
        assert before ^ after, "Incorrect usage"
        return after ^ self.normalize_before

    def reorder_incremental_state(
        self, incremental_state: Dict[str, Tensor], new_order: Tensor
    ):
        self.self_attn.reorder_incremental_state(incremental_state, new_order)
        self.encoder_attn.reorder_incremental_state(incremental_state, new_order)
        self.conv.reorder_incremental_state(incremental_state, new_order)

    def extra_repr(self):
        return (
            "dropout={}, relu_dropout={}, input_dropout={}, normalize_before={}".format(
                self.dropout,
                self.relu_dropout,
                self.input_dropout,
                self.normalize_before,
            )
        )


class ConvDecoderConfig(ConfigBase):
    dropout: float = 0.1
    decoder_embed_dim: int = 128
    decoder_input_dim: int = 128
    decoder_output_dim: int = 128
    max_target_positions: int = 128
    decoder_learned_pos: bool = False
    no_token_positional_embeddings: bool = False
    positional_embedding_type: PostionalEmbedType = PostionalEmbedType.LEARNED
    combine_pos_embed: PostionalEmbedCombine = PostionalEmbedCombine.CONCAT
    decoder_normalize_before: bool = False


class LightConvDecoderBase(PyTextIncrementalDecoderComponent):
    class Config(ModuleConfig):
        decoder_config: ConvDecoderConfig = ConvDecoderConfig()
        layer_config: LightConvDecoderLayer.Config = LightConvDecoderLayer.Config()
        decoder_kernel_size_list: List[int] = [3, 7, 15]

    @classmethod
    def from_config(cls, config, tgt_dict, tgt_embedding):
        kernel_size_list = config.decoder_kernel_size_list
        layers = []
        for size in kernel_size_list:
            assert (
                config.decoder_config.decoder_embed_dim
                == config.layer_config.decoder_embed_dim
            )
            layers.append(create_module(config.layer_config, kernel_size=size))
        return cls(tgt_dict, tgt_embedding, layers, config.decoder_config)

    def __init__(self, target_dict, embed_tokens, layers, decoder_config):
        super().__init__()
        self.dropout = decoder_config.dropout

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = decoder_config.decoder_embed_dim
        output_embed_dim = decoder_config.decoder_output_dim

        padding_idx = target_dict.get_pad_index()
        self.max_target_positions = decoder_config.max_target_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)  # todo: try with input_embed_dim
        self.padding_idx = padding_idx

        self.no_token_positional_embeddings = (
            decoder_config.no_token_positional_embeddings
        )
        # creating this is also conditional
        self.project_in_dim = (
            Linear(input_embed_dim, embed_dim)
            if embed_dim != input_embed_dim
            else PlaceholderIdentity()
        )
        self.embed_layer_norm = LayerNorm(embed_dim)
        self.combine_pos_embed = decoder_config.combine_pos_embed.value
        self.embed_positions = build_positional_embedding(
            positional_embedding_type=decoder_config.positional_embedding_type,
            combine_pos_embed=decoder_config.combine_pos_embed,
            max_target_positions=decoder_config.max_target_positions,
            input_embed_dim=input_embed_dim,
            embed_dim=embed_dim,
            padding_idx=padding_idx,
            no_token_positional_embeddings=decoder_config.no_token_positional_embeddings,
        )

        self.layers = nn.ModuleList(layers)

        self.project_out_dim = (
            Linear(embed_dim, output_embed_dim, bias=False)
            if embed_dim != output_embed_dim
            else PlaceholderIdentity()
        )

        self.normalize = decoder_config.decoder_normalize_before
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = PlaceholderIdentity()

    def forward_unprojected(
        self,
        prev_output_tokens: Tensor,
        encoder_out: Dict[str, Tensor],
        incremental_state: Optional[Dict[str, Tensor]] = None,
        timestep: Optional[int] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        output_dict: Dict[str, Tensor] = {}
        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens([[prev_output_tokens]])

        if not self.no_token_positional_embeddings:
            # TODO : Verify incremental generation for AR mode
            x = self.pos_embed(x, prev_output_tokens)
        else:
            x = self.project_in_dim(x)

        x = self.embed_layer_norm(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        output_dict["decoder_layer_0"] = x.clone()

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        last_layer_attn: Optional[Tensor] = None

        decoder_padding_mask = prev_output_tokens.eq(self.padding_idx)
        target_lengths = (~decoder_padding_mask).sum(dim=1)

        if not decoder_padding_mask.any():
            decoder_mask = None
        else:
            decoder_mask = decoder_padding_mask

        encoder = encoder_out["encoder_out"]
        encoder_mask: Optional[Tensor] = None
        if "encoder_mask" in encoder_out:
            encoder_mask = encoder_out["encoder_mask"]

        # decoder layers
        for idx, layer in enumerate(self.layers):
            encoder = encoder_out["encoder_out"]
            encoder_mask: Optional[Tensor] = None
            if "encoder_mask" in encoder_out:
                encoder_mask = encoder_out["encoder_mask"]
            x, last_layer_attn = layer(
                x, encoder, encoder_mask, decoder_mask, incremental_state
            )
            output_dict["decoder_layer_" + str(idx + 1)] = x.transpose(0, 1).clone()

        if self.normalize:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        x = self.project_out_dim(x)

        if last_layer_attn is not None:
            output_dict["attn_scores"] = last_layer_attn
        output_dict["target_lengths"] = target_lengths
        output_dict["decoder_mask"] = decoder_padding_mask

        for key in encoder_out.keys():
            output_dict[key] = encoder_out[key]

        return x, output_dict

    def pos_embed(self, x, src_tokens):
        # TODO : Positional embeddings needs to be tested in AR mode
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
        """Maximum output length supported by the decoder."""
        if self.no_token_positional_embeddings:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def reorder_incremental_state(
        self, incremental_state: Dict[str, Tensor], new_order: Tensor
    ):
        for layer in self.layers:
            layer.reorder_incremental_state(incremental_state, new_order)

    def get_probs(
        self, decoder_out: Tuple[Tensor, Dict[str, Tensor]]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        return self.projection_layer.get_probs(decoder_out)


class LightConvDecoder(LightConvDecoderBase):
    def __init__(self, target_dict, embed_tokens, layers, decoder_config):
        super().__init__(target_dict, embed_tokens, layers, decoder_config)
        self.projection_layer = DecoderWithLinearOutputProjection(
            target_dict, target_dict, decoder_config.decoder_output_dim
        )

    def forward(
        self,
        prev_output_tokens: Tensor,
        encoder_out: Dict[str, Tensor],
        incremental_state: Optional[Dict[str, Tensor]] = None,
        timestep: Optional[int] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        hidden_decoder_output = self.forward_unprojected(
            prev_output_tokens, encoder_out, incremental_state, timestep
        )
        return self.projection_layer(
            encoder_out=encoder_out,
            decoder_out=hidden_decoder_output,
            incremental_state=incremental_state,
        )

    def get_probs(
        self, decoder_out: Tuple[Tensor, Dict[str, Tensor]]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        return self.projection_layer.get_probs(decoder_out)


class LightConvDecoupledDecoder(LightConvDecoderBase):
    class Config(ModuleConfig):
        decoder_config: ConvDecoderConfig = ConvDecoderConfig()
        layer_config: LightConvDecoderLayer.Config = LightConvDecoderLayer.Config()
        decoder_kernel_size_list: List[int] = [3, 7, 15]
        decoder_layers: int = 3
        decoupled_attention_heads: int = 1
        ontology_generation_only: bool = False
        model_output_logprob: bool = True

    def __init__(
        self,
        target_dict,
        embed_tokens,
        layers,
        decoder_config,
        ontology_generation_only,
        decoupled_attention_heads,
        model_output_logprob,
    ):
        super().__init__(target_dict, embed_tokens, layers, decoder_config)
        fixed_generation_vocab = None
        if ontology_generation_only:
            fixed_generation_vocab = extract_ontology_vocab(target_dict)
        self.projection_layer = DecoupledDecoderHead(
            target_dict,
            target_dict,
            out_embed_dim=decoder_config.decoder_output_dim,
            encoder_hidden_dim=decoder_config.decoder_input_dim,
            pointer_attention_heads=decoupled_attention_heads,
            fixed_generation_vocab=fixed_generation_vocab,
            model_output_logprob=model_output_logprob,
        )

    def forward(
        self,
        prev_output_tokens: Tensor,
        encoder_out: Dict[str, Tensor],
        incremental_state: Optional[Dict[str, Tensor]] = None,
        timestep: Optional[int] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        hidden_decoder_output = self.forward_unprojected(
            prev_output_tokens, encoder_out, incremental_state, timestep
        )
        return self.projection_layer(
            encoder_out=encoder_out,
            decoder_out=hidden_decoder_output,
            incremental_state=incremental_state,
        )

    @classmethod
    def from_config(cls, config, tgt_dict, tgt_embedding):
        kernel_size_list = config.decoder_kernel_size_list
        layers = []

        config.layer_config.decoder_embed_dim = log_and_overwrite(
            param_name="layer_config.decoder_embed_dim, decoder_config.decoder_embed_dim",
            x=config.layer_config.decoder_embed_dim,
            y=config.decoder_config.decoder_embed_dim,
        )

        config.decoder_config.decoder_input_dim = log_and_overwrite(
            param_name="decoder_config.decoder_input_dim, decoder_config.decoder_embed_dim",
            x=config.decoder_config.decoder_input_dim,
            y=config.decoder_config.decoder_embed_dim,
        )

        config.layer_config.decoder_conv_dim = log_and_overwrite(
            param_name="layer_config.decoder_conv_dim, decoder_config.decoder_embed_dim",
            x=config.layer_config.decoder_conv_dim,
            y=config.decoder_config.decoder_embed_dim,
        )

        for size in kernel_size_list:
            assert (
                config.decoder_config.decoder_embed_dim
                == config.layer_config.decoder_embed_dim
            )
            layers.append(create_module(config.layer_config, kernel_size=size))
        return cls(
            tgt_dict,
            tgt_embedding,
            layers,
            config.decoder_config,
            config.ontology_generation_only,
            config.decoupled_attention_heads,
            config.model_output_logprob,
        )

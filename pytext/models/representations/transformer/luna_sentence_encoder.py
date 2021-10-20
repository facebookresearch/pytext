#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import math
from typing import Callable, Optional

import torch
from fairseq import utils
from fairseq.modules import LayerNorm, PositionalEmbedding, LayerDropModuleList
from fairseq.modules.fairseq_dropout import FairseqDropout
from pytext.utils.usage import log_class_usage
from torch import nn
from torch.nn import Parameter

from . import LunarMultiheadAttention


class LunaSentenceEncoderLayer(nn.Module):
    """
    Implements a Luna Encoder Layer used in masked pre-trained language models
    and fine-tuned classication/regression mdoel
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        num_projected_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        normalize_before: bool = False,
        tie_kv=True,
        export: bool = False,
        init_fn: Callable = None,
    ) -> None:
        super().__init__()

        if init_fn is not None:
            init_fn()

        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )
        self.activation_dropout_module = FairseqDropout(
            activation_dropout, module_name=self.__class__.__name__
        )

        # Initialize blocks
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.self_attn = self.build_self_attention(
            self.embedding_dim,
            num_attention_heads,
            num_projected_attention_heads,
            dropout=attention_dropout,
            tie_kv=tie_kv,
        )

        # layer norm associated with the self attention layer
        self.normalize_before = normalize_before
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim, export=export)
        self.self_atten_proj_layer_norm = LayerNorm(self.embedding_dim, export=export)

        self.fc1 = self.build_fc1(
            self.embedding_dim,
            ffn_embedding_dim,
        )
        self.fc2 = self.build_fc2(
            ffn_embedding_dim,
            self.embedding_dim,
        )

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = LayerNorm(self.embedding_dim, export=export)

    def build_fc1(self, input_dim, output_dim):
        return nn.Linear(input_dim, output_dim)

    def build_fc2(self, input_dim, output_dim):
        return nn.Linear(input_dim, output_dim)

    def build_self_attention(
        self,
        embed_dim,
        num_attention_heads,
        num_projected_attention_heads,
        dropout,
        tie_kv,
        q_noise,
        qn_block_size,
    ):
        return LunarMultiheadAttention(
            embed_dim,
            num_attention_heads,
            num_projected_attention_heads,
            dropout=dropout,
            self_attention=True,
            tie_kv=tie_kv,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )

    def forward(
        self,
        x: torch.Tensor,
        px: torch.Tensor,
        x_padding_mask: Optional[torch.Tensor] = None,
        px_padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        residual = x
        presidual = px
        # apply prev layer norm
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
            px = self.self_atten_proj_layer_norm(px)

        x, px, attn = self.self_attn(
            query=x,
            pquery=px,
            context=x,
            context_padding_mask=x_padding_mask,
            pcontext_padding_mask=px_padding_mask,
            need_weights=False,
        )
        # apply dropout
        x = self.dropout_module(x)
        px = self.dropout_module(px)
        # residual
        x = residual + x
        px = presidual + px

        # apply post layer norm
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)
            px = self.self_atten_proj_layer_norm(px)

        #######################################################################
        # Feed-Forward Network
        residual = x
        # apply prev layer norm
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x

        # apply post layer norm
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x, px, attn


def init_bert_params(module):
    """
    Initialize the weights specific to the BERT Model.
    This overrides the default initializations depending on the specified arguments.
        1. If normal_init_linear_weights is set then weights of linear
           layer will be initialized using the normal distribution and
           bais will be set to the specified value.
        2. If normal_init_embed_weights is set then weights of embedding
           layer will be initialized using the normal distribution.
        3. If normal_init_proj_weights is set then weights of
           in_project_weight for MultiHeadAttention initialized using
           the normal distribution (to be validated).
    """

    if isinstance(module, LunaSentenceEncoder):
        module.projected_embeddings.data.normal_(mean=0.0, std=0.02)
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, LunarMultiheadAttention):
        module.pq_proj.weight.data.normal_(mean=0.0, std=0.02)
        module.q_proj.weight.data.normal_(mean=0.0, std=0.02)
        if module.pc_proj is not None:
            module.pc_proj.weight.data.normal_(mean=0.0, std=0.02)
            module.c_proj.weight.data.normal_(mean=0.0, std=0.02)
        else:
            module.pk_proj.weight.data.normal_(mean=0.0, std=0.02)
            module.pv_proj.weight.data.normal_(mean=0.0, std=0.02)
            module.k_proj.weight.data.normal_(mean=0.0, std=0.02)
            module.v_proj.weight.data.normal_(mean=0.0, std=0.02)


class LunaSentenceEncoder(nn.Module):
    """
    Implementation for a Bi-directional Luna based Sentence Encoder used
    in masked pre-trained language models.
    This first computes the token embedding using the token embedding matrix,
    position embeddings (if specified) and segment embeddings
    (if specified). After applying the specified number of
    TransformerEncoderLayers, it outputs all the internal states of the
    encoder as well as the final representation associated with the first
    token (usually CLS token).
    Input:
        - tokens: B x T matrix representing sentences
        - segment_labels: B x T matrix representing segment label for tokens
    Output:
        - a tuple of the following:
            - a list of internal model states used to compute the
              predictions where each tensor has shape T x B x C
            - sentence representation associated with first input token
              in format B x C.
    """

    def __init__(
        self,
        padding_idx: int,
        vocab_size: int,
        projection_length: int = 128,
        num_encoder_layers: int = 12,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 12,
        num_projected_attention_heads: int = 12,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        layerdrop: float = 0.0,
        max_seq_len: int = 512,
        num_segments: int = 0,
        use_position_embeddings: bool = True,
        offset_positions_by_padding: bool = True,
        layernorm_embedding: bool = False,
        normalize_before: bool = False,
        dynamic_projection: bool = True,
        tie_kv=False,
        apply_bert_init: bool = False,
        activation_fn: str = "gelu",
        learned_pos_embedding: bool = True,
        embed_scale: float = None,
        freeze_embeddings: bool = False,
        n_trans_layers_to_freeze: int = 0,
        export: bool = False,
        traceable: bool = False,
    ) -> None:

        super().__init__()
        self.padding_idx = padding_idx
        self.vocab_size = vocab_size
        self.proj_len = projection_length
        self.dynamic_projection = dynamic_projection
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )
        self.layerdrop = layerdrop
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self.num_segments = num_segments
        self.use_position_embeddings = use_position_embeddings
        self.apply_bert_init = apply_bert_init
        self.learned_pos_embedding = learned_pos_embedding
        self.traceable = traceable
        self.tpu = False  # whether we're on TPU

        self.embed_tokens = self.build_embedding(
            self.vocab_size, self.embedding_dim, self.padding_idx
        )
        self.embed_scale = embed_scale

        if self.num_segments > 0:
            self.segment_embeddings = nn.Embedding(
                self.num_segments, self.embedding_dim, padding_idx=None
            )
            nn.init.normal_(
                self.segment_embeddings.weight, mean=0.0, std=self.embedding_dim ** -0.5
            )
        else:
            self.segment_embeddings = None

        self.embed_positions = (
            PositionalEmbedding(
                self.max_seq_len,
                self.embedding_dim,
                padding_idx=(self.padding_idx if offset_positions_by_padding else None),
                learned=self.learned_pos_embedding,
            )
            if self.use_position_embeddings
            else None
        )

        self.projected_embeddings = Parameter(
            torch.Tensor(self.proj_len, self.embedding_dim)
        )
        nn.init.normal_(
            self.projected_embeddings, mean=0.0, std=self.embedding_dim ** -0.5
        )
        if self.use_position_embeddings and not self.learned_pos_embedding:
            projected_positions = get_sinusoidal_positional_embedding(
                self.proj_len, self.embedding_dim
            )
            if self.embed_scale is None:
                self.embed_scale = math.sqrt(self.embedding_dim)
        else:
            projected_positions = None
        self.register_buffer("projected_positions", projected_positions)

        if self.layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.layerdrop)
        else:
            self.layers = nn.ModuleList([])

        self.layers.extend(
            [
                self.build_luna_sentence_encoder_layer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=ffn_embedding_dim,
                    num_attention_heads=num_attention_heads,
                    num_projected_attention_heads=num_projected_attention_heads,
                    dropout=self.dropout_module.p,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=activation_fn,
                    normalize_before=normalize_before,
                    tie_kv=tie_kv,
                    export=export,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        assert not layernorm_embedding or not normalize_before

        if layernorm_embedding:
            self.emb_layer_norm = LayerNorm(self.embedding_dim, export=export)
            self.proj_emb_layer_norm = LayerNorm(self.embedding_dim, export=export)
        else:
            self.emb_layer_norm = None
            self.proj_emb_layer_norm = None

        if normalize_before:
            self.layer_norm = LayerNorm(self.embedding_dim, export=export)
            self.proj_layer_norm = LayerNorm(self.embedding_dim, export=export)
        else:
            self.layer_norm = None
            self.proj_layer_norm = None

        # Apply initialization of model params after building the model
        if self.apply_bert_init:
            self.apply(init_bert_params)

        def freeze_module_params(m):
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = False

        if freeze_embeddings:
            self.projected_embeddings.requires_grad = False
            freeze_module_params(self.embed_tokens)
            freeze_module_params(self.segment_embeddings)
            freeze_module_params(self.embed_positions)
            freeze_module_params(self.emb_layer_norm)
            freeze_module_params(self.proj_emb_layer_norm)

        for layer in range(n_trans_layers_to_freeze):
            freeze_module_params(self.layers[layer])

        log_class_usage(__class__)

    def build_embedding(self, vocab_size, embedding_dim, padding_idx):
        embed_tokens = nn.Embedding(vocab_size, embedding_dim, padding_idx)
        nn.init.normal_(embed_tokens.weight, mean=0, std=embedding_dim ** -0.5)
        return embed_tokens

    def build_luna_sentence_encoder_layer(
        self,
        embedding_dim,
        ffn_embedding_dim,
        num_attention_heads,
        num_projected_attention_heads,
        dropout,
        attention_dropout,
        activation_dropout,
        activation_fn,
        normalize_before,
        tie_kv,
        export,
        q_noise,
        qn_block_size,
    ):
        return LunaSentenceEncoderLayer(
            embedding_dim=embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            num_attention_heads=num_attention_heads,
            num_projected_attention_heads=num_projected_attention_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            activation_fn=activation_fn,
            normalize_before=normalize_before,
            tie_kv=tie_kv,
            export=export,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )

    def prepare_for_tpu_(self, **kwargs):
        self.tpu = True

    def forward(
        self,
        tokens: torch.Tensor,
        segment_labels: torch.Tensor = None,
        last_state_only: bool = False,
        positions: Optional[torch.Tensor] = None,
    ):

        # compute padding mask. This is needed for multi-head attention
        # B x T
        x_padding_mask = tokens.eq(self.padding_idx)
        lengths = tokens.size(1) - x_padding_mask.sum(1)
        max_len = lengths.max() if self.dynamic_projection else self.proj_len

        x = self.embed_tokens(tokens)
        px = self.projected_embeddings[:max_len]

        if self.embed_scale is not None:
            x *= self.embed_scale
            px *= self.embed_scale

        if self.embed_positions is not None:
            x += self.embed_positions(tokens, positions=positions)
        if self.projected_positions is not None:
            px += self.projected_positions[:max_len]

        if self.segment_embeddings is not None and segment_labels is not None:
            x += self.segment_embeddings(segment_labels)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)
            px = self.proj_emb_layer_norm(px)

        bsz = x.size(0)
        len, dim = px.size()
        # L x C -> B x L x C
        px = px.unsqueeze(0).expand(bsz, len, dim)

        if self.dynamic_projection:
            pidx = torch.arange(len).unsqueeze(0).to(x.device)
            # B x L
            px_padding_mask = pidx.ge(lengths.unsqueeze(1))
        else:
            px_padding_mask = None

        if not self.traceable and not self.tpu:
            if not x_padding_mask.any():
                x_padding_mask = None
            if px_padding_mask is not None and not px_padding_mask.any():
                px_padding_mask = None

        x = self.dropout_module(x)
        px = self.dropout_module(px)

        # account for padding while computing the representation
        if x_padding_mask is not None:
            x = x * (1 - x_padding_mask.unsqueeze(-1).type_as(x))
        if px_padding_mask is not None:
            px = px * (1 - px_padding_mask.unsqueeze(-1).type_as(px))

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        # B x L x C -> L x B x C
        px = px.transpose(0, 1)

        inner_states = []
        if not last_state_only:
            inner_states.append(x)

        for layer in self.layers:
            x, px, _ = layer(
                x, px, x_padding_mask=x_padding_mask, px_padding_mask=px_padding_mask
            )
            if not last_state_only:
                inner_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)
            px = self.proj_layer_norm(px)

        # sentence_cls_rep = x[0, :, :]
        # sentence_proj_rep = px

        if last_state_only:
            inner_states = [x]

        return inner_states


def get_sinusoidal_positional_embedding(length, embed_dim):
    half_dim = embed_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
    emb = torch.arange(length, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(length, -1)
    if embed_dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros(length, 1)], dim=1)
    return emb

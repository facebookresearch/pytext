#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


from typing import List

import numpy as np
import torch
from pytext.models.representations.transformer import (
    TransformerLayer,
    MultiheadSelfAttention,
)
from pytext.models.roberta import RoBERTaEncoder
from torch import nn, Tensor

torch.ops.load_library("//pytorch/FasterTransformers3.1:faster_transformers")


@torch.jit.script
def sequence_mask(neg_padding_mask: Tensor) -> Tensor:
    neg_padding_mask = neg_padding_mask.half()
    mask = neg_padding_mask.view(
        neg_padding_mask.size(0), 1, 1, neg_padding_mask.size(1)
    )
    m2 = mask.transpose(2, 3)
    return mask * m2


def to_fast_transformer_weights(layer):
    attn_qw, attn_kw, attn_vw = layer.attention.input_projection.weight.chunk(3, dim=0)
    attn_qb, attn_kb, attn_vb = layer.attention.input_projection.bias.chunk(3, dim=0)
    attn_ow = layer.attention.output_projection.weight
    attn_ob = layer.attention.output_projection.bias
    attn_nw = layer.attention_layer_norm.weight
    attn_nb = layer.attention_layer_norm.bias
    inter_w = layer.residual_mlp.mlp.__getattr__("0").weight
    inter_b = layer.residual_mlp.mlp.__getattr__("0").bias
    output_w = layer.residual_mlp.mlp.__getattr__("3").weight
    output_b = layer.residual_mlp.mlp.__getattr__("3").bias
    norm_w = layer.final_layer_norm.weight
    norm_b = layer.final_layer_norm.bias
    fast_transformer_weights = [
        attn_qw.transpose(-1, -2).contiguous(),
        attn_qb,
        attn_kw.transpose(-1, -2).contiguous(),
        attn_kb,
        attn_vw.transpose(-1, -2).contiguous(),
        attn_vb,
        attn_ow.transpose(-1, -2).contiguous(),
        attn_ob,
        attn_nw,
        attn_nb,
        inter_w.transpose(-1, -2).contiguous(),
        inter_b,
        output_w.transpose(-1, -2).contiguous(),
        output_b,
        norm_w,
        norm_b,
        torch.tensor(0),
    ]
    return [t.half().cuda() for t in fast_transformer_weights]


# Hide custom class behind torch.jit.script as jit.trace doesn't support custom classes.
class NVTransformerStack(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
        self.layer_num = len(layers)

    def forward(self, encoded: Tensor, neg_padding_mask: Tensor) -> List[Tensor]:
        # seq_lengths: [B,]
        sequence_lengths = neg_padding_mask.sum(dim=1, dtype=torch.int32)
        # Note - this does a HtoD copy/stream synchronization
        # Necessary because our implementation does not handle the zero-token case.
        if sequence_lengths.sum().item() == 0:
            return [encoded.transpose(0, 1)] + [
                torch.zeros_like(encoded.transpose(0, 1)) for _ in range(self.layer_num)
            ]

        # Note - this also does a HtoD copy/stream synchronization.
        (
            hidden_states,
            sequence_id_offset,
        ) = torch.ops.fastertransformer.build_mask_remove_padding(
            encoded, sequence_lengths
        )

        # attention_mask: [B, 1, T, T]
        attention_mask = sequence_mask(neg_padding_mask).half()
        # trt_seq_len: [B + 1,]
        trt_seq_len = torch.cumsum(
            torch.cat(
                [
                    torch.zeros(
                        1, device=sequence_lengths.device, dtype=sequence_lengths.dtype
                    ),
                    sequence_lengths,
                ],
                dim=0,
            ),
            dim=0,
            dtype=torch.int32,
        )

        all_hidden_states = [hidden_states]
        for i in range(self.layer_num):
            hidden_states = self.layers[i].forward(
                hidden_states, attention_mask, trt_seq_len, sequence_id_offset
            )
            all_hidden_states.append(hidden_states)

        # Remap back to padded [B, T, D] representation, and transpose to [T, B, D].
        states = []
        for hidden_states in all_hidden_states:
            # Ideally jit.tracer will eliminate unncessary ones as the corresponding
            # output tensor would be unused. It doesn't seem to currently, though.
            state = torch.ops.fastertransformer.rebuild_padding(
                hidden_states, sequence_id_offset, attention_mask, 0
            )
            # Convert to [T, B, D] representation.
            states.append(state.transpose(1, 0))
        return states


class NVFasterTransformerEncoder(nn.Module):
    def __init__(self, old_transformer):
        super().__init__()
        remove_padding = True
        use_trt_kernel = True
        allow_gemm_test = False
        int8_mode = 0
        self.layer_num = len(old_transformer.layers)
        self.int8_mode = int8_mode
        self.token_embedding = old_transformer.token_embedding
        self.positional_embedding = old_transformer.positional_embedding
        self.embedding_layer_norm = old_transformer.embedding_layer_norm
        self.dropout = old_transformer.dropout
        self.padding_idx = old_transformer.padding_idx
        num_headss, scalings, embed_dims = set(), set(), set()
        for layer in old_transformer.layers:
            assert isinstance(layer, TransformerLayer)
            att = layer.attention
            assert isinstance(att, MultiheadSelfAttention)
            num_headss.add(att.num_heads)
            scalings.add(att.scaling)
            embed_dims.add(att.embed_dim)
            # TODO: ResidualMLP check.

        # ensure values match.
        (num_heads,) = num_headss
        (scaling,) = scalings
        (embed_dims,) = embed_dims
        head_dim = embed_dims // num_heads
        np.testing.assert_allclose(scaling, 1.0 / np.sqrt(head_dim))
        encoders = []
        for i in range(self.layer_num):
            encoders.append(
                torch.classes.FasterTransformer.Encoder(
                    *to_fast_transformer_weights(old_transformer.layers[i]),
                    num_heads,
                    head_dim,
                    remove_padding,
                    int8_mode,
                    self.layer_num,
                    i,
                    allow_gemm_test,
                    use_trt_kernel
                )
            )

        self.encoder = torch.jit.script(NVTransformerStack(encoders))

    def forward(self, tokens: Tensor) -> List[Tensor]:
        # Vanilla transformer prelude
        neg_padding_mask = tokens.ne(self.padding_idx)
        embedded = self.token_embedding(tokens)
        embedded_positions = self.positional_embedding(tokens)
        normed = self.embedding_layer_norm(embedded + embedded_positions)
        normed = self.dropout(normed)
        padded_normed = normed * neg_padding_mask.unsqueeze(-1)

        # encoded: [B, T, C]
        encoded = padded_normed.half()
        states = self.encoder(encoded, neg_padding_mask)

        # commonly you can retrieve a single "sentence representation" as
        # states[-1].transpose(0, 1)
        return states


# Swap a transformer for only RoBERTaEncoder encoders
def swap_modules_for_faster_transformer(model):
    if hasattr(model, "encoder") and isinstance(model.encoder, RoBERTaEncoder):
        old_transformer = model.encoder.encoder.transformer
        model.encoder.encoder.transformer = NVFasterTransformerEncoder(old_transformer)
        return model
    else:
        return model

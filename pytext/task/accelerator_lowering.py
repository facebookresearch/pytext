#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List

import torch
from accelerators.pytorch.lib.glow_decorator import accelerator
from pytext.models.roberta import RoBERTaEncoder
from torch import nn


@accelerator([("NNPI", {"NNPI_IceCores": "12", "NNPINumParallelChunks": "12"})])
class AcceleratorTransformerLayers(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def forward(
        self, encoded: torch.Tensor, padding_mask: torch.Tensor
    ) -> List[torch.Tensor]:
        states = [encoded]

        for layer in self.layers:
            encoded = layer(encoded, padding_mask)
            states.append(encoded)

        return states


# Special reimplementation of transformer which separates the
# layers into a separate module for easy lowering to accelerator
class AcceleratorTransformer(nn.Module):
    def __init__(self, transformer):
        super().__init__()
        self.padding_idx = transformer.padding_idx
        self.token_embedding = transformer.token_embedding
        self.layers = AcceleratorTransformerLayers(transformer.layers)
        self.positional_embedding = transformer.positional_embedding
        self.embedding_layer_norm = transformer.embedding_layer_norm
        self.dropout = transformer.dropout

    def forward(self, tokens: torch.Tensor) -> List[torch.Tensor]:
        # compute padding mask. This is needed for multi-head attention
        padding_mask = tokens.eq(self.padding_idx)

        embedded = self.token_embedding(tokens)
        embedded_positions = self.positional_embedding(tokens)

        normed = self.embedding_layer_norm(embedded + embedded_positions)
        normed = self.dropout(normed)
        # account for padding while computing the representation
        padded_normed = normed * (1 - padding_mask.unsqueeze(-1).type_as(normed))

        # B x T x C -> T x B x C
        encoded = padded_normed.transpose(0, 1)

        states = self.layers(encoded, padding_mask)
        return states


# Swap a transformer for only RoBERTaEncoder encoders
def swap_modules_for_accelerator(model):
    if hasattr(model, "encoder") and isinstance(model.encoder, RoBERTaEncoder):
        old_transformer = model.encoder.encoder.transformer
        model.encoder.encoder.transformer = AcceleratorTransformer(old_transformer)
        return model
    else:
        return model


def lower_modules_to_accelerator(
    model, trace, seq_padding_control, batch_padding_control
):
    import torch_glow

    if hasattr(model, "encoder") and isinstance(model.encoder, RoBERTaEncoder):
        backend = "NNPI"
        submod_path, compilation_spec_dict = accelerator.get_modules(model, backend)[0]
        embedding_dim = model.encoder.encoder.transformer.token_embedding.embedding_dim
        spec = torch_glow.CompilationSpec()
        spec.get_settings().set_glow_backend(backend)
        compilation_group = torch_glow.CompilationGroup()
        spec.compilation_groups_append(compilation_group)
        compilation_group_settings = compilation_group.get_settings()
        compilation_group_settings.set_convert_to_fp16(True)
        for k, v in compilation_spec_dict.items():
            compilation_group.get_settings().backend_specific_opts_insert(k, v)

        for seq_len in seq_padding_control:
            if seq_len <= 0:
                continue
            for batch_size in batch_padding_control:
                if batch_size <= 0:
                    continue
                input1 = torch.randn(
                    [seq_len, batch_size, embedding_dim], dtype=torch.float32
                )
                input2 = torch.randn([batch_size, seq_len]).bool()
                input_specs = torch_glow.input_specs_from_tensors([input1, input2])
                compilation_group.input_sets_append(input_specs)

        trace = torch_glow.to_glow_selective(
            trace,
            {f"model.{submod_path}": spec},
            inplace=False,
        )

        return trace
    else:
        return trace

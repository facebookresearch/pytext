#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Iterable, List, Tuple

import torch

accelerator_lowering_supported = True
try:
    from accelerators.pytorch.lib.glow_decorator import accelerator
except ImportError:
    accelerator_lowering_supported = False

    from .nop_decorator import accelerator

    print("Accelerator Lowering not supported!")

from pytext.config import ExportConfig
from pytext.models.representations.bilstm import BiLSTM
from pytext.models.roberta import RoBERTaEncoder
from pytext.utils.usage import log_accelerator_feature_usage
from torch import nn


def accelerator_transformerLayers_inputs(
    model: nn.Module,
    trace: torch.jit.ScriptFunction,
    export_options: ExportConfig,
    dataset_iterable: Iterable,
    module_path,
):
    import torch_glow

    # we use the padding control from the Export Config:
    if export_options is None:
        export_options = ExportConfig()

    if export_options.seq_padding_control is None:
        raise RuntimeError("seq padding control not specified")
    if export_options.batch_padding_control is None:
        raise RuntimeError("batch padding control not specified")

    batch_padding_control = export_options.batch_padding_control

    # Restrict seq_padding_control to valid ranges
    seq_padding_control = []
    max_seq_len = trace.get_max_seq_len()
    for pad in export_options.seq_padding_control:
        if pad < max_seq_len:
            seq_padding_control.append(pad)
    seq_padding_control.append(max_seq_len)

    # this should use a method, or module_path, instead of being hardcoded
    # embedding_dim = model.encoder.encoder.transformer.token_embedding.embedding_dim
    embedding_dim = accelerator.get_embedding_module_from_path(model, module_path)

    input_examples = []
    for seq_len in seq_padding_control:
        if seq_len <= 0:
            continue
        for batch_size in batch_padding_control:
            if batch_size <= 0:
                continue
            # Todo: We directly generate data input instead of using dataset_iterable, enhance later
            input1 = torch.randn(
                [seq_len, batch_size, embedding_dim], dtype=torch.float32
            )
            input2 = torch.randn([batch_size, seq_len]).bool()
            input_specs = torch_glow.input_specs_from_tensors([input1, input2])
            input_examples.append(input_specs)

    return input_examples


class AcceleratorTransformerLayersInternal(nn.Module):
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


# accelerator imported from .nop_decorator to avoid ImportError when glow_decorator is not available
@accelerator(
    [
        (
            "NNPI",
            {
                "NNPI_IceCores": "12",
                "NNPINumParallelChunks": "12",
                "NNPIUseGeluLUT": "true",
                "glow:ConvertToFP16": "true",
            },
        ),
        (
            "NNPI:throughput_optimized",
            {
                "NNPI_IceCores": "4",
                "NNPINumParallelChunks": "4",
                "NNPIUseGeluLUT": "true",
                "glow:ConvertToFP16": "true",
                "glow:ReplicationCount": "3",
            },
        ),
    ],
    inputs_function=accelerator_transformerLayers_inputs,
)
class AcceleratorTransformerLayers(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = AcceleratorTransformerLayersInternal(layers)

    def forward(
        self, encoded: torch.Tensor, padding_mask: torch.Tensor
    ) -> List[torch.Tensor]:
        return self.layers(encoded, padding_mask)


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


def accelerator_lstm_inputs(
    model: nn.Module,
    trace: torch.jit.ScriptFunction,
    export_options: ExportConfig,
    dataset_iterable: Iterable,
    module_path,
):
    import torch_glow

    # we use the padding control from the Export Config:
    if export_options is None:
        export_options = ExportConfig()

    if export_options.seq_padding_control is None:
        raise RuntimeError("seq padding control not specified")
    if export_options.batch_padding_control is None:
        raise RuntimeError("batch padding control not specified")

    batch_padding_control = export_options.batch_padding_control
    seq_padding_control = export_options.seq_padding_control
    embedding_dim = trace.embedding.word_embedding.embedding_dim * 2
    lstm_num_layers = trace.lstm_num_layers
    lstm_dim = trace.lstm_dim

    input_examples = []
    for seq_len in seq_padding_control:
        if seq_len <= 0:
            continue
        for batch_size in batch_padding_control:
            if batch_size <= 0:
                continue
            # Todo: We directly generate data input instead of using dataset_iterable, enhance later
            input_embedding = torch.randn(
                [batch_size, seq_len, embedding_dim], dtype=torch.float32
            )
            input_hidden = torch.randn(
                [batch_size, lstm_num_layers, lstm_dim], dtype=torch.float32
            )
            input_cell = torch.randn(
                [batch_size, lstm_num_layers, lstm_dim], dtype=torch.float32
            )
            input_specs = torch_glow.input_specs_from_tensors(
                [input_embedding, input_hidden, input_cell]
            )
            input_examples.append(input_specs)

    return input_examples


@accelerator(
    [("NNPI", {"NNPI_IceCores": "1", "NNPINumParallelChunks": "12"})],
    inputs_function=accelerator_lstm_inputs,
)
class AcceleratorLSTMLayers(nn.Module):
    def __init__(self, lstm):
        super().__init__()
        self.lstm = lstm
        self.num_layers = lstm.num_layers
        self.hidden_size = lstm.hidden_size
        self.lstm.batch_first = False  # NNPI only support batch_first = false

    def forward(
        self, lstm_input: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor
    ):
        lstm_input = lstm_input.transpose(0, 1)
        hidden = hidden.transpose(0, 1)
        cell = cell.transpose(0, 1)
        rep, new_state = self.lstm(lstm_input, (hidden, cell))
        return rep, new_state[0], new_state[1]


class AcceleratorBiLSTM(nn.Module):
    def __init__(self, biLSTM):
        super().__init__()
        self.dropout = biLSTM.dropout
        self.pack_sequence = biLSTM.pack_sequence
        self.disable_sort_in_jit = biLSTM.disable_sort_in_jit
        self.lstm = AcceleratorLSTMLayers(biLSTM.lstm)
        self.representation_dim = biLSTM.representation_dim
        self.padding_value = biLSTM.padding_value

    def forward(
        self,
        embedded_tokens: torch.Tensor,
        seq_lengths: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        rep, new_hidden, new_cell = self.lstm(embedded_tokens, states[0], states[1])
        new_hidden = new_hidden.reshape(
            self.lstm.num_layers, rep.size(1), self.lstm.hidden_size
        ).transpose(0, 1)
        new_cell = new_cell.reshape(
            self.lstm.num_layers, rep.size(1), self.lstm.hidden_size
        ).transpose(0, 1)
        rep = rep.transpose(0, 1)
        return rep, (new_hidden, new_cell)


def lower_modules_to_accelerator(
    model: nn.Module, trace, export_options: ExportConfig, throughput_optimize=False
):
    # Raise error if accelerator could not be imported
    if not accelerator_lowering_supported:
        raise RuntimeError("Accelerator Lowering not supported!")

    import torch_glow

    log_accelerator_feature_usage("build.NNPI")
    if (
        (hasattr(model, "encoder") and isinstance(model.encoder, RoBERTaEncoder))
        or (
            hasattr(model, "representation")
            and isinstance(model.representation, AcceleratorBiLSTM)
        )
        or (
            hasattr(model, "lower_module")
            # Internal CNN LM module to add accelerator support.
            and type(model.lower_module).__qualname__ == "CNNLowerModule"
        )
    ):
        backend = "NNPI"
        backend_qualifier = ""

        if throughput_optimize:
            backend_qualifier = ":throughput_optimized"

        modules_to_lower = accelerator.get_modules(model, backend + backend_qualifier)

        if len(modules_to_lower) < 1:
            raise RuntimeError("Need at least one module to lower to accelerator")
        elif len(modules_to_lower) > 1:
            print(f"Warning. Received {len(modules_to_lower)} modules to lower.")
            print("Warning. Only lowering first module.")

        (
            submod_modelpath,
            compilation_spec_dict,
            inputs_function,
        ) = modules_to_lower[0]
        submod_tracepath = accelerator.model2trace_path(submod_modelpath)
        spec = torch_glow.CompilationSpec()
        spec.get_settings().set_glow_backend(backend)
        compilation_group = torch_glow.CompilationGroup()
        spec.compilation_groups_append(compilation_group)
        compilation_group_settings = compilation_group.get_settings()

        # Set values from dict that are not set via backend-specific opts
        compilation_group_settings.set_convert_to_fp16(
            compilation_spec_dict.pop("glow:ConvertToFP16", "true") in ["true", "True"]
        )
        compilation_group_settings.set_replication_count(
            int(compilation_spec_dict.pop("glow:ReplicationCount", "1"))
        )

        for k, v in compilation_spec_dict.items():
            compilation_group.get_settings().backend_specific_opts_insert(k, v)

        if inputs_function is not None:
            input_sets = inputs_function(
                model, trace, export_options, None, submod_modelpath
            )
        else:
            raise RuntimeError(
                "inputs_function needs to be specified in accelerator decorator"
            )
        compilation_group.set_input_sets(input_sets)

        trace = torch_glow.to_glow_selective(
            trace,
            {submod_tracepath: spec},
            inplace=False,
        )

        return trace
    else:
        return trace


def nnpi_rewrite_roberta_transformer(model):
    model.encoder.encoder.transformer = AcceleratorTransformer(
        model.encoder.encoder.transformer
    )


def nnpi_rewrite_bilstm(model):
    model.representation = AcceleratorBiLSTM(model.representation)

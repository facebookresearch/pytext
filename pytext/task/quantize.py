#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
from pytext.models.roberta import RoBERTaEncoder
from torch.quantization import HistogramObserver, QConfig, default_weight_observer


def quantize_statically(
    model, inputs, data_loader, linear_only=False, module_swap=False
):
    if (
        hasattr(model, "encoder")
        and isinstance(model.encoder, RoBERTaEncoder)
        and linear_only
    ):
        qconfig = QConfig(
            activation=HistogramObserver.with_args(reduce_range=False),
            weight=default_weight_observer,
        )
        qconfig_dict = {"": None}
        if module_swap:
            layers = model.encoder.encoder.transformer.layers.layers
            layers_str = "encoder.encoder.transformer.layers.layers"
        else:
            layers = model.encoder.encoder.transformer.layers
            layers_str = "encoder.encoder.transformer.layers"

        # skip first layer
        for layer_idx in range(1, len(layers)):
            qconfig_dict[
                layers_str + ".{}.attention.input_projection".format(layer_idx)
            ] = qconfig
            qconfig_dict[
                layers_str + ".{}.attention.output_projection".format(layer_idx)
            ] = qconfig
            for mlp_idx, m in enumerate(layers[layer_idx].residual_mlp.mlp):
                # Only quantize first linear otherwise there are accuarcy issues
                if type(m) == torch.nn.Linear and mlp_idx < 1:
                    qconfig_dict[
                        layers_str
                        + ".{}.residual_mlp.mlp.{}".format(layer_idx, mlp_idx)
                    ] = qconfig
        trace = model.graph_mode_quantize(
            inputs, data_loader, qconfig_dict=qconfig_dict, force_quantize=True
        )
    else:
        trace = model.graph_mode_quantize(inputs, data_loader)

    return trace

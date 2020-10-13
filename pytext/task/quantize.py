#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
from torch.quantization import HistogramObserver, QConfig, default_weight_observer

def quantize_statically(model, data_loader, linear_only=False):
    if isinstance(model.encoder, RoBERTaEncoder) and linear_only:
        qconfig = QConfig(
            activation=HistogramObserver.with_args(reduce_range=False),
            weight=default_weight_observer,
        )
        qconfig_dict = {"": None}
        for layer_idx in range(len(model.encoder.encoder.transformer.layers)):
            qconfig_dict[
                "encoder.encoder.transformer.layers.{}.attention.input_projection".format(
                    layer_idx
                )
            ] = qconfig
            qconfig_dict[
                "encoder.encoder.transformer.layers.{}.attention.output_projection".format(
                    layer_idx
                )
            ] = qconfig
            for mlp_idx, m in enumerate(
                model.encoder.encoder.transformer.layers[layer_idx].residual_mlp.mlp
            ):
                if type(m) == torch.nn.Linear:
                    qconfig_dict[
                        "encoder.encoder.transformer.layers.{}.residual_mlp.mlp.{}".format(
                            layer_idx, mlp_idx
                        )
                    ] = qconfig
        trace = model.graph_mode_quantize(
            inputs, data_loader, qconfig_dict=qconfig_dict
        )
    else:
        trace = model.graph_mode_quantize(inputs, data_loader)

    return trace

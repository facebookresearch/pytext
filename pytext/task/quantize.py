#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
from pytext.models.roberta import RoBERTaEncoder
from pytext.utils.usage import log_accelerator_feature_usage, log_feature_usage
from torch.quantization import (
    HistogramObserver,
    QConfig,
    default_weight_observer,
    per_channel_dynamic_qconfig,
)
from torch.quantization.quantize_fx import prepare_fx, convert_fx

# Quantize linear layers using fx static or dynamic quantization
def quantize_fx(model, inputs, data_loader, dynamic=True, selective=False):

    if hasattr(model, "encoder") and isinstance(model.encoder, RoBERTaEncoder):

        static = not dynamic

        if dynamic:
            qconfig = per_channel_dynamic_qconfig
        else:
            qconfig = QConfig(
                activation=HistogramObserver.with_args(reduce_range=False),
                weight=default_weight_observer,
            )

        # Only linear layers
        qconfig_dict = {"": None}
        if static and selective:
            qconfig_dict["module_name"] = []
            layers = model.encoder.encoder.transformer.layers.layers.layers
            layers_str = "layers"
            # skip first layer
            for layer_idx in range(1, len(layers)):
                qconfig_dict["module_name"].append(
                    (
                        layers_str + ".{}.attention.input_projection".format(layer_idx),
                        qconfig,
                    )
                )
                qconfig_dict["module_name"].append(
                    (
                        layers_str
                        + ".{}.attention.output_projection".format(layer_idx),
                        qconfig,
                    )
                )
                for mlp_idx, m in enumerate(layers[layer_idx].residual_mlp.mlp):
                    # Only quantize first linear otherwise there are accuarcy issues with static quantization
                    if type(m) == torch.nn.Linear and mlp_idx < 1:
                        qconfig_dict["module_name"].append(
                            (
                                layers_str
                                + ".{}.residual_mlp.mlp.{}".format(layer_idx, mlp_idx),
                                qconfig,
                            )
                        )
        else:
            qconfig_dict["object_type"] = [(torch.nn.Linear, qconfig)]

        def calibrate(model, loader, max_samples=-1):
            model.eval()
            with torch.no_grad():
                for (idx, d) in enumerate(loader):
                    print("Running sample input #" + str(idx))
                    model(d[1]["tokens"])
                    if idx == max_samples:
                        break

        prepared_model = prepare_fx(
            model.encoder.encoder.transformer.layers.layers, qconfig_dict
        )  # fuse modules and insert observers

        model.encoder.encoder.transformer.layers.layers = prepared_model
        if static:
            calibrate(model, data_loader)  # run calibration on sample data
        model.encoder.encoder.transformer.layers.layers = convert_fx(prepared_model)

        # Trace the submodule in order to fix the interface
        if static:
            input1 = torch.randn([2, 1, 1024], dtype=torch.float)
            input2 = torch.randn([1, 2]).bool()
            traced = torch.jit.trace(
                model.encoder.encoder.transformer.layers.layers, (input1, input2)
            )
            model.encoder.encoder.transformer.layers.layers = traced

        # Trace the overall module
        trace = model.trace(inputs)

        return trace


def quantize_statically(
    model, inputs, data_loader, linear_only=False, module_swap=False
):
    log_feature_usage("export.quantize.statically")
    if (
        hasattr(model, "encoder")
        and isinstance(model.encoder, RoBERTaEncoder)
        and linear_only
    ):
        log_accelerator_feature_usage("quantize.statically")
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

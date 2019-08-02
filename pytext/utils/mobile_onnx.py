#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import itertools

import numpy as np
import onnx
import torch
from caffe2.python import core, utils, workspace
from caffe2.python.onnx import backend as caffe2_backend
from pytext.utils.onnx import convert_caffe2_blob_name


def create_context(init_net):
    workspace.ResetWorkspace()
    assert workspace.RunNetOnce(init_net)


def pytorch_to_caffe2(
    model,
    export_input,
    external_input_names,
    output_names,
    export_path,
    export_onnx_path=None,
):
    num_tensors = 0
    for inp in export_input:
        num_tensors += len(inp) if isinstance(inp, (tuple, list)) else 1
    assert len(external_input_names) == num_tensors
    all_input_names = external_input_names[:]
    for name, _ in model.named_parameters():
        all_input_names.append(name)
    # export the pytorch model to ONNX
    if export_onnx_path:
        print(f"Saving onnx model to: {export_onnx_path}")
    else:
        export_onnx_path = export_path
    model.eval()
    with torch.no_grad():
        torch.onnx.export(
            model,
            export_input,
            export_onnx_path,
            input_names=all_input_names,
            output_names=output_names,
            export_params=True,
        )
    onnx_model = onnx.load(export_onnx_path)
    onnx.checker.check_model(onnx_model)
    # split onnx model into init_net and predict_net
    init_net, predict_net = caffe2_backend.Caffe2Backend.onnx_graph_to_caffe2_net(
        onnx_model
    )
    return (init_net, predict_net)


def add_feats_numericalize_ops(init_net, predict_net, vocab_map, input_names):
    init_net, final_predict_net, final_input_names = get_numericalize_net(
        init_net, predict_net, vocab_map, input_names
    )

    create_context(init_net)
    init_net = init_net.Proto()
    final_predict_net.Proto().op.extend(predict_net.op)
    predict_net = final_predict_net.Proto()
    return (init_net, predict_net, final_input_names)


def get_numericalize_net(init_net, predict_net, vocab_map, input_names):
    init_net = core.Net(init_net)
    final_input_names = input_names.copy()

    create_context(init_net)
    vocab_indices = create_vocab_indices_map(init_net, vocab_map)

    final_predict_net = core.Net(predict_net.name + "_processed")
    final_inputs = set(
        {ext_input for ext_input in input_names if ext_input not in vocab_map.keys()}
    )

    for ext_input in final_inputs:
        final_predict_net.AddExternalInput(ext_input)

    # add external_input and external_output from init_net
    init_net_proto = init_net.Proto()
    items = set(
        {
            item
            for item in itertools.chain(
                init_net_proto.external_input, init_net_proto.external_output
            )
        }
    )
    for item in items:
        final_predict_net.AddExternalInput(item)

    for feat in vocab_map.keys():
        raw_input_blob = final_predict_net.AddExternalInput(
            convert_caffe2_blob_name(feat)
        )
        flattened_input_blob = final_predict_net.FlattenToVec(raw_input_blob)
        flattened_ids = final_predict_net.IndexGet(
            [vocab_indices[feat], flattened_input_blob]
        )
        final_predict_net.ResizeLike([flattened_ids, raw_input_blob], [feat])
        final_input_names[input_names.index(feat)] = convert_caffe2_blob_name(feat)

    return (init_net, final_predict_net, final_input_names)


def create_vocab_indices_map(init_net, vocab_map):
    vocab_indices = {}
    for feat_name, vocab in vocab_map.items():
        assert len(vocab) > 1
        vocab_indices[feat_name] = create_vocab_index(
            np.array(vocab, dtype=str)[1:], init_net, workspace, feat_name + "_index"
        )
    return vocab_indices


def create_vocab_index(vocab_list, net, net_workspace, index_name):
    vocab_blob = net.AddExternalInput(f"{index_name}_vocab")

    net.GivenTensorStringFill(
        [],
        [str(vocab_blob)],
        arg=[
            utils.MakeArgument("shape", vocab_list.shape),
            utils.MakeArgument("values", vocab_list),
        ],
    )

    vocab_index = net.StringIndexCreate([], index_name)
    net.IndexLoad([vocab_index, vocab_blob], [vocab_index], skip_first_entry=0)
    net.IndexFreeze([vocab_index], [vocab_index])

    return vocab_index

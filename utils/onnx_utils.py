#!/usr/bin/env python3

from caffe2.python.onnx import backend as caffe2_backend
# TODO @shicong figure out why open source version is not working
import caffe2.caffe2.fb.predictor.predictor_exporter as pe
# import caffe2.python.predictor.predictor_exporter as pe
from caffe2.python import workspace, core

import onnx
import torch
import numpy as np


def pytorch_to_caffe2(
    model, export_input, external_input_names, output_names, export_path
):
    all_input_names = external_input_names[:]
    for name, _ in model.named_parameters():
        all_input_names.append(name)
    # # export the pytorch model to ONNX
    torch.onnx.export(
        model,
        export_input,
        export_path,
        input_names=all_input_names,
        output_names=output_names,
        export_params=True,
    )
    onnx_model = onnx.load(export_path)
    onnx.checker.check_model(onnx_model)
    # Convert the ONNX model to a caffe2 net
    c2_prepared = caffe2_backend.prepare(onnx_model)
    return c2_prepared


def create_vocab_index(vocab_list, net, net_workspace, index_name):
    vocab_index = net.StringIndexCreate([], index_name)
    vocab_blob = net.AddExternalInput(net.NextName())
    net_workspace.FeedBlob(str(vocab_blob), vocab_list)
    # Populates the index with all the vocab
    net.IndexGet([vocab_index, vocab_blob])
    # Freeze the index to not add any other words in runtime
    net.IndexFreeze([vocab_index], [vocab_index])
    return vocab_index


def add_feats_numericalize_ops(c2_prepared, vocab_map, input_names):
    predict_net = c2_prepared.predict_net  # Protobuf of the predict_net
    init_net = core.Net(c2_prepared.init_net)
    final_input_names = input_names.copy()
    with c2_prepared.workspace._ctx:
        vocab_indices = {}
        for feat_name, vocab in vocab_map.items():
            assert len(vocab) > 1
            vocab_indices[feat_name] = create_vocab_index(
                # Skip index 0 as it is reserved for unkwon tokens
                # in Caffe2's index implementation
                np.array(vocab[1:], dtype=str),
                init_net,
                c2_prepared.workspace,
                feat_name + "_index",
            )
        # Add operators to convert string features to ids based on the vocab
        final_predict_net = core.Net(c2_prepared.predict_net.name + "_processed")
        final_inputs = set(
            {
                ext_input
                for ext_input in predict_net.external_input
                if ext_input not in vocab_map.keys()
            }
        )
        for ext_input in final_inputs:
            final_predict_net.AddExternalInput(ext_input)

        for feat in vocab_map.keys():
            # Caffe2 predictor appends the ":value" suffix to any feature that
            # contains a string list https://fburl.com/rviadm83
            raw_input_blob = final_predict_net.AddExternalInput(feat + "_str:value")
            # IndexGet expects flat tensors, so flatten the batch first then
            # Resize it back after the lookup
            flattened_input_blob = final_predict_net.FlattenToVec(raw_input_blob)
            flattened_ids = final_predict_net.IndexGet(
                [vocab_indices[feat], flattened_input_blob]
            )
            final_predict_net.ResizeLike([flattened_ids, raw_input_blob], [feat])
            final_input_names[input_names.index(feat)] = feat + "_str:value"
        # Copy over the other list of the ops
        final_predict_net.Proto().op.extend(predict_net.op)
        # Update predict_net and init_net
        c2_prepared.predict_net = final_predict_net.Proto()
        c2_prepared.init_net = init_net.Proto()

    return c2_prepared, final_input_names


def export_nets_to_predictor_file(
    c2_prepared, input_names, output_names, predictor_path
):
    # netdef external_input includes internally produced blobs
    actual_external_inputs = set()
    produced = set()
    for operator in c2_prepared.predict_net.op:
        for blob in operator.input:
            if blob not in produced:
                actual_external_inputs.add(blob)
        for blob in operator.output:
            produced.add(blob)
    for blob in output_names:
        if blob not in produced:
            actual_external_inputs.add(blob)
    param_names = [blob for blob in actual_external_inputs if blob not in input_names]

    init_net = core.Net(c2_prepared.init_net)
    predict_net = core.Net(c2_prepared.predict_net)

    # Required because of https://github.com/pytorch/pytorch/pull/6456/files
    with c2_prepared.workspace._ctx:
        workspace.RunNetOnce(init_net)
        predictor_export_meta = pe.PredictorExportMeta(
            predict_net=predict_net,
            parameters=param_names,
            inputs=input_names,
            outputs=output_names,
            shapes={x: () for x in input_names + output_names},
            net_type="simple",
        )
        pe.save_to_db(
            db_type="log_file_db",
            db_destination=predictor_path,
            predictor_export_meta=predictor_export_meta,
        )

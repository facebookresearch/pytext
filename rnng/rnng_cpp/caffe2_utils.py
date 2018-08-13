#!/usr/bin/env python3

from typing import Dict, List

from caffe2.python import core, dyndep, workspace
from pytext.config import PyTextConfig, config_from_json
import caffe2.caffe2.fb.predictor.predictor_exporter as pe
import numpy as np
import torch

RNNG_VERSION = 0
RNNG_INPUT_NAMES = [  # Prediction Request from www, with default values
    "tokens_vals_str:value",  # string_list
    "dict_vals_str:value",  # string_list
    "dict_weights",
    "dict_lens",
]
RNNG_INPUT_TYPES = [
    np.array(""),
    np.array(""),
    np.array([0.], dtype=np.float32),
    np.array([0], dtype=np.int_),
]
RNNG_OUTPUTS = ["actions", "tokens", "scores", "pretty_print"]


def read_pytorch_model(model_path: str) -> Dict:
    snapshot = torch.load(model_path, map_location=lambda storage, loc: storage)
    model_state = snapshot["model_state"]
    pytext_config = config_from_json(PyTextConfig, snapshot["pytext_config"])
    oracle_dicts = snapshot["oracle_dicts"]

    model_config = get_model_config(pytext_config)

    config = {
        "model_config": model_config,
        "model_state": model_state,
        "weight_names": list(model_state.keys()),
        "actions_vec": oracle_dicts.actions_bidict.get_sorted_objs(),
        "terminals_vec": oracle_dicts.terminal_bidict.get_sorted_objs(),
        "dictfeats_vec": oracle_dicts.dictfeat_bidict.get_sorted_objs()
        if oracle_dicts.dictfeat_bidict is not None
        else [],  # [] because we cannot pass NULL to C++ Parser.
    }
    return config


def get_model_config(pytext_config) -> List[float]:
    jobspec = pytext_config.jobspec
    rnng_config = pytext_config.jobspec.model
    model_config = [
        RNNG_VERSION,
        rnng_config.lstm.lstm_dim,
        rnng_config.lstm.num_layers,
        jobspec.features.word_feat.embed_dim,
        rnng_config.max_open_NT,
        jobspec.features.dict_feat.embed_dim,
        rnng_config.dropout,
    ]
    return [float(mc) for mc in model_config]


def export_model(pytorch_model_path: str, caffe2_model_path: str):
    workspace.GlobalInit(["caffe2", "--caffe2_log_level=2"])
    dyndep.InitOpsLibrary("//pytext/rnng/rnng_cpp:rnng_op")

    pytorch_config = read_pytorch_model(pytorch_model_path)
    for weight_name, weight_value in pytorch_config["model_state"].items():
        workspace.FeedBlob(weight_name, weight_value.detach().numpy())

    for input_name, input_type in zip(RNNG_INPUT_NAMES, RNNG_INPUT_TYPES):
        workspace.FeedBlob(input_name, input_type)

    all_inputs = RNNG_INPUT_NAMES + pytorch_config["weight_names"]

    predict_net = core.Net("prediction_net")
    predict_net.RNNGParser(
        all_inputs,
        RNNG_OUTPUTS,
        model_config=pytorch_config["model_config"],
        weight_names=pytorch_config["weight_names"],
        actions_vec=pytorch_config["actions_vec"],
        terminals_vec=pytorch_config["terminals_vec"],
        dictfeats_vec=pytorch_config["dictfeats_vec"],
    )

    workspace.RunNetOnce(predict_net)

    predictor_export_meta = pe.PredictorExportMeta(
        predict_net=predict_net, parameters=all_inputs, inputs=[], outputs=RNNG_OUTPUTS
    )

    pe.save_to_db(
        db_type="log_file_db",
        db_destination=caffe2_model_path,
        predictor_export_meta=predictor_export_meta,
    )

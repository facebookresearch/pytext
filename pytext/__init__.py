#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import uuid

import numpy as np
from caffe2.python import workspace
from caffe2.python.predictor import predictor_exporter

from .builtin_task import register_builtin_tasks
from .config.component import create_featurizer
from .data.featurizer import InputRecord
from .utils.onnx_utils import CAFFE2_DB_TYPE, convert_caffe2_blob_name


register_builtin_tasks()


def _predict(workspace_id, feature_config, predict_net, featurizer, input):
    workspace.SwitchWorkspace(workspace_id)
    features = featurizer.featurize(InputRecord(**input))
    if feature_config.word_feat:
        for blob_name in feature_config.word_feat.export_input_names:
            converted_blob_name = convert_caffe2_blob_name(blob_name)
            workspace.blobs[converted_blob_name] = np.array(
                [features.tokens], dtype=str
            )
        workspace.blobs["tokens_lens"] = np.array([len(features.tokens)], dtype=np.int_)
    if feature_config.dict_feat:
        dict_feats, weights, lens = feature_config.dict_feat.export_input_names
        converted_dict_blob_name = convert_caffe2_blob_name(dict_feats)
        workspace.blobs[converted_dict_blob_name] = np.array(
            [features.gazetteer_feats], dtype=str
        )
        workspace.blobs[weights] = np.array(
            [features.gazetteer_feat_weights], dtype=np.float32
        )
        workspace.blobs[lens] = np.array(features.gazetteer_feat_lengths, dtype=np.int_)

    if feature_config.char_feat:
        for blob_name in feature_config.char_feat.export_input_names:
            converted_blob_name = convert_caffe2_blob_name(blob_name)
            workspace.blobs[converted_blob_name] = np.array(
                [features.characters], dtype=str
            )

    workspace.RunNet(predict_net)
    return {
        str(blob): workspace.blobs[blob][0] for blob in predict_net.external_outputs
    }


def create_predictor(config, model_file=None):
    workspace_id = str(uuid.uuid4())
    workspace.SwitchWorkspace(workspace_id, True)
    predict_net = predictor_exporter.prepare_prediction_net(
        filename=model_file or config.export_caffe2_path, db_type=CAFFE2_DB_TYPE
    )

    task = config.task
    feature_config = task.features
    featurizer = create_featurizer(task.featurizer, feature_config)

    return lambda input: _predict(
        workspace_id, feature_config, predict_net, featurizer, input
    )

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import json
import logging
import uuid
from functools import lru_cache
from typing import Callable, Mapping, Optional

import numpy as np
from caffe2.python import workspace
from caffe2.python.predictor import predictor_exporter
from pytext.data.sources.data_source import DataSource
from pytext.task import load
from pytext.task.new_task import NewTask
from pytext.utils.file_io import PathManager, register_http_url_handler
from pytext.workflow import _set_cuda

from .builtin_task import register_builtin_tasks
from .config import PyTextConfig, pytext_config_from_json
from .utils.onnx import CAFFE2_DB_TYPE, convert_caffe2_blob_name


register_builtin_tasks()
register_http_url_handler()


Predictor = Callable[[Mapping[str, str]], Mapping[str, np.array]]


def _predict(workspace_id, predict_net, model, tensorizers, input):
    workspace.SwitchWorkspace(workspace_id)
    tensor_dict = {
        name: tensorizer.prepare_input(input)
        for name, tensorizer in tensorizers.items()
    }
    model_inputs = model.arrange_caffe2_model_inputs(tensor_dict)
    model_input_names = model.get_export_input_names(tensorizers)
    vocab_to_export = model.vocab_to_export(tensorizers)
    for blob_name, model_input in zip(model_input_names, model_inputs):
        converted_blob_name = blob_name
        dtype = np.float32
        if blob_name in vocab_to_export:
            converted_blob_name = convert_caffe2_blob_name(blob_name)
            dtype = str

        workspace.blobs[converted_blob_name] = np.array([model_input], dtype=dtype)
    workspace.RunNet(predict_net)
    return {
        str(blob): workspace.blobs[blob][0] for blob in predict_net.external_outputs
    }


def load_config(filename: str) -> PyTextConfig:
    """
    Load a PyText configuration file from a file path.
    See pytext.config.pytext_config for more info on configs.
    """
    with PathManager.open(filename) as file:
        config_json = json.loads(file.read())
    if "config" not in config_json:
        return pytext_config_from_json(config_json)
    return pytext_config_from_json(config_json["config"])


def create_predictor(
    config: PyTextConfig,
    model_file: Optional[str] = None,
    db_type: str = CAFFE2_DB_TYPE,
    task: Optional[NewTask] = None,
    cache_size: int = 0,
) -> Predictor:
    """
    Create a simple prediction API from a training config and an exported caffe2
    model file. This model file should be created by calling export on a trained
    model snapshot.
    """
    workspace_id = str(uuid.uuid4())
    workspace.SwitchWorkspace(workspace_id, True)
    predict_net = predictor_exporter.prepare_prediction_net(
        filename=model_file or config.export_caffe2_path, db_type=db_type
    )

    new_task = task or NewTask.from_config(config.task)
    input_tensorizers = {
        name: tensorizer
        for name, tensorizer in new_task.data.tensorizers.items()
        if tensorizer.is_input
    }

    def predict_fn(input):
        return _predict(
            workspace_id, predict_net, new_task.model, input_tensorizers, input
        )

    if cache_size < 0:
        return lru_cache(maxsize=None)(predict_fn)
    elif cache_size > 0:
        return lru_cache(maxsize=cache_size)(predict_fn)
    else:
        return predict_fn


def batch_predict_caffe2_model(
    pytext_model_file: str,
    caffe2_model_file: str,
    db_type: str = CAFFE2_DB_TYPE,
    data_source: Optional[DataSource] = None,
    use_cuda=False,
    task: Optional[NewTask] = None,
    train_config: Optional[PyTextConfig] = None,
    cache_size: int = 0,
):
    """
    Gets predictions from caffe2 model from a batch of examples.

    Args:
        pytext_model_file: Path to pytext model file (required if task and
            training config is not specified)
        caffe2_model_file: Path to caffe2 model file
        db_type: DB type to use for caffe2
        data_source: Data source for test examples
        use_cuda: Whether to turn on cuda processing
        task: The pytext task object
        train_config: The pytext training config
        cache_size: The LRU cache size to use for prediction. 0 = no cache,
            -1 = boundless cache, [1, inf) = size of cache
    """
    logging.info(f"Loading data processing config from {pytext_model_file}")

    _set_cuda(use_cuda)
    if task is None or train_config is None:
        task, train_config, _ = load(pytext_model_file)

    data_source = data_source or task.data.data_source
    logging.info("Loading Caffe2 model")
    predictor = create_predictor(
        train_config, caffe2_model_file, db_type, task, cache_size
    )
    logging.info(f"Model loaded, start testing")
    predictions = [predictor(example) for example in data_source.test]
    return predictions

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os

import torch
from pytext.config import PyTextConfig, config_to_json, pytext_config_from_json
from pytext.data import CommonMetadata
from pytext.models import Model

from .task import create_task


DATA_STATE = "data_state"
CONFIG_JSON = "config_json"
MODEL_STATE = "model_state"
SERIALIZE_VERSION_KEY = "pytext_serialization_version"

LATEST_SERIALIZE_VERSION = 2


def save(config: PyTextConfig, model: Model, meta: CommonMetadata) -> None:
    """
    Save a task, will save the original config, model state and metadata
    """
    save_path = config.save_snapshot_path
    print(f"Saving pytorch model to: {save_path}")
    model.save_modules(base_path=config.modules_save_dir)
    state = {
        DATA_STATE: meta,
        CONFIG_JSON: config_to_json(PyTextConfig, config),
        MODEL_STATE: model,
        SERIALIZE_VERSION_KEY: LATEST_SERIALIZE_VERSION,
    }
    torch.save(state, save_path)


def load(load_path: str):
    """
    Load task, will construct the task using the saved config then load metadata
    and model state.
    """
    if not (load_path and os.path.isfile(load_path)):
        raise ValueError(f"Invalid snapshot path{load_path}")
    print(f"Loading model from {load_path}...")
    state = torch.load(load_path, map_location=lambda storage, loc: storage)
    if SERIALIZE_VERSION_KEY not in state:
        return load_v1(state)
    else:
        return LOADER_VERSION_MAP[state[SERIALIZE_VERSION_KEY]](state)


def load_v1(state):
    config = pytext_config_from_json(state[CONFIG_JSON])

    task = create_task(
        config.task, metadata=state[DATA_STATE], model_state=state[MODEL_STATE]
    )
    return task, config


def load_v2(state):
    config = pytext_config_from_json(state[CONFIG_JSON])
    model = state[MODEL_STATE]
    task = create_task(config.task, metadata=state[DATA_STATE], model=model)
    return task, config


LOADER_VERSION_MAP = {1: load_v1, 2: load_v2}

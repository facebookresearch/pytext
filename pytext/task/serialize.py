#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
from typing import Dict

import torch
from pytext.config import PyTextConfig, config_to_json, pytext_config_from_json
from pytext.data import CommonMetadata
from pytext.data.tensorizers import Tensorizer
from pytext.models import Model

from .task import create_task


DATA_STATE = "data_state"
CONFIG_JSON = "config_json"
MODEL_STATE = "model_state"
SERIALIZE_VERSION_KEY = "pytext_serialization_version"
TENSORIZERS = "tensorizers"

LATEST_SERIALIZE_VERSION = 0
LOADER_VERSION_MAP = {}


def register_snapshot_loader(version):
    def decorator(fn):
        LOADER_VERSION_MAP[version] = fn
        global LATEST_SERIALIZE_VERSION
        LATEST_SERIALIZE_VERSION = max(LATEST_SERIALIZE_VERSION, version)
        return fn

    return decorator


def save(
    config: PyTextConfig,
    model: Model,
    meta: CommonMetadata,
    tensorizers: Dict[str, Tensorizer],
) -> None:
    """
    Save a task, will save the original config, model state and metadata
    """
    save_path = config.save_snapshot_path
    print(f"Saving pytorch model to: {save_path}")
    model.save_modules(base_path=config.modules_save_dir)
    state = {
        DATA_STATE: meta,
        CONFIG_JSON: config_to_json(PyTextConfig, config),
        MODEL_STATE: model.state_dict(),
        SERIALIZE_VERSION_KEY: LATEST_SERIALIZE_VERSION,
        TENSORIZERS: tensorizers,
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


@register_snapshot_loader(1)
def load_v1(state):
    config = pytext_config_from_json(state[CONFIG_JSON])

    task = create_task(
        config.task, metadata=state[DATA_STATE], model_state=state[MODEL_STATE]
    )
    return task, config


@register_snapshot_loader(2)
def load_v2(state):
    config = pytext_config_from_json(state[CONFIG_JSON])
    model_state = state[MODEL_STATE]
    tensorizers = state[TENSORIZERS]
    task = create_task(
        config.task,
        metadata=state[DATA_STATE],
        model_state=model_state,
        tensorizers=tensorizers,
    )
    return task, config

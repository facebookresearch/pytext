#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import io
import os
from typing import Dict, Optional

import torch
from pytext.config import PyTextConfig, config_to_json, pytext_config_from_json
from pytext.data import CommonMetadata
from pytext.data.tensorizers import Tensorizer
from pytext.models import Model
from pytext.trainers.training_state import TrainingState


DATA_STATE = "data_state"
CONFIG_JSON = "config_json"
MODEL_STATE = "model_state"
SERIALIZE_VERSION_KEY = "pytext_serialization_version"
TENSORIZERS = "tensorizers"
TRAINING_STATE = "training_state"


LATEST_SERIALIZE_VERSION = 3
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
    meta: Optional[CommonMetadata],
    tensorizers: Dict[str, Tensorizer],
    training_state: Optional[TrainingState] = None,
    f: Optional[io.IOBase] = None,
) -> None:
    """
    Save all stateful information of a training task to a specified file-like
    object, will save the original config, model state, metadata,
    training state if training is not completed
    """
    if config.modules_save_dir:
        model.save_modules(base_path=config.modules_save_dir)

    # Currently torch.save() has error pickling certain models when not saving
    # by modelstate_dict(), thus currently overriding the model in
    # training_state with None. and put back aftwards
    # https://github.com/pytorch/pytorch/issues/15116
    model_in_training_state = None
    if training_state:
        model_in_training_state, training_state.model = training_state.model, None
    try:
        state = {
            DATA_STATE: meta,
            CONFIG_JSON: config_to_json(PyTextConfig, config),
            MODEL_STATE: model.state_dict(),
            SERIALIZE_VERSION_KEY: LATEST_SERIALIZE_VERSION,
            TENSORIZERS: tensorizers,
            TRAINING_STATE: training_state,
        }
        if f is None:
            save_path = config.save_snapshot_path
            print(f"Saving pytorch model to: {save_path}")
            torch.save(state, save_path)
        else:
            torch.save(state, f)
    finally:
        if training_state:
            training_state.model = model_in_training_state


def load(load_path: str):
    """
    Load task, config and training state from a saved snapshot, will construct
    the task using the saved config then load metadata and model state.
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
    # importing in file level generates circular import/dependency failures,
    # that need refator later to fix
    from .task import create_task

    task = create_task(
        config.task, metadata=state[DATA_STATE], model_state=state[MODEL_STATE]
    )
    return task, config


@register_snapshot_loader(2)
def load_v2(state):
    config = pytext_config_from_json(state[CONFIG_JSON])
    model_state = state[MODEL_STATE]
    tensorizers = state[TENSORIZERS]
    # importing in file level generates circular import/dependency failures,
    # that need refator later to fix
    from .task import create_task

    task = create_task(
        config.task,
        metadata=state[DATA_STATE],
        model_state=model_state,
        tensorizers=tensorizers,
    )
    return task, config


@register_snapshot_loader(3)
def load_v3(state):
    config = pytext_config_from_json(state[CONFIG_JSON])
    # importing in file level generates circular import/dependency failures,
    # that need refator later to fix
    from .task import create_task

    task = create_task(
        config.task, metadata=state[DATA_STATE], model_state=state[MODEL_STATE]
    )
    training_state = state[TRAINING_STATE]
    return task, config, training_state

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import io
import os
from typing import Dict, List, Optional

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
    model_state = state[MODEL_STATE]
    training_state = state[TRAINING_STATE]
    if training_state and training_state.tensorizers:
        tensorizers = training_state.tensorizers
    else:
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
    return task, config, training_state


def load_checkpoint(f: io.IOBase):
    state = torch.load(f, map_location=lambda storage, loc: storage)
    if SERIALIZE_VERSION_KEY not in state:
        return load_v1(state)
    else:
        return LOADER_VERSION_MAP[state[SERIALIZE_VERSION_KEY]](state)


def save_checkpoint(
    f: io.IOBase,
    config: PyTextConfig,
    model: Model,
    meta: Optional[CommonMetadata],
    tensorizers: Dict[str, Tensorizer],
    training_state: Optional[TrainingState] = None,
) -> str:
    # Currently torch.save() has error pickling certain models when not saving
    # by model.state_dict(), thus currently overriding the model in
    # training_state with None, and put back saving
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
        torch.save(state, f)
    finally:
        if training_state:
            training_state.model = model_in_training_state


def get_latest_checkpoint_path() -> str:
    """
    Return most recent saved checkpoint path in str
    Returns: checkpoint_path (str)
    """
    return _CHECKPOINT_MANAGER.get_latest_checkpoint_path()


def get_post_training_snapshot_path() -> str:
    return _CHECKPOINT_MANAGER.get_post_training_snapshot_path()


class CheckpointManager:
    """
        CheckpointManager is class abstraction to manage training job's
        checkpoints with different IO and storage, using two functions:
        save() and load().
    """

    def __init__(self):
        # keep a list of saved checkpoint path
        self._saved_paths: List[str] = []
        self._post_training_snapshot_path = None

    # generate per epoch checkpoint save path
    def generate_checkpoint_path(self, config: PyTextConfig, identifier: str):
        dir_name = os.path.dirname(config.save_snapshot_path)
        return "{}/checkpoint-{}".format(dir_name, identifier)

    def save(
        self,
        config: PyTextConfig,
        model: Model,
        meta: Optional[CommonMetadata],
        tensorizers: Dict[str, Tensorizer],
        training_state: Optional[TrainingState] = None,
        identifier: str = None,
    ) -> str:
        """
        save a checkpoint to given path, config, model and training_state
        together represent the checkpoint. When identifier is None, this
        function is used to save post-training snapshot
        """
        if identifier:
            # saving during-training checkpoints
            save_path = self.generate_checkpoint_path(config, identifier)
            print("Saving checkpoint to ", save_path)
        else:
            # saving post-training snapshot if no identifer given
            save_path = config.save_snapshot_path
            print(f"Saving pytorch model to: {save_path}")

        with open(save_path, "wb") as checkpoint_f:
            saved_path = save_checkpoint(
                checkpoint_f, config, model, meta, tensorizers, training_state
            )
            if identifier:
                self._saved_paths.append(saved_path)
            else:
                self._post_training_snapshot_path = saved_path
        return save_path

    def load(self, load_path: str):
        """
        Loads a checkpoint from disk.
        Args:
            load_path (str): the file path to load for checkpoint
        Returns: task (Task), config (PyTextConfig) and training_state (TrainingState)
        """
        if not (load_path and os.path.isfile(load_path)):
            raise ValueError(f"Invalid snapshot path{load_path}")
        print(f"Loading model from {load_path}...")
        with open(load_path, "rb") as checkpoint_f:
            return load_checkpoint(checkpoint_f)

    def list(self) -> List[str]:
        """
        Return all existing checkpoint path in str
        Returns: checkpoint_path_list (List[str]), list elements are in the same
        order of checkpoint saving
        """
        return self._saved_paths

    def get_latest_checkpoint_path(self) -> str:
        """
        Return most recent saved checkpoint path in str
        Returns: checkpoint_path (str)
        """
        return self._saved_paths[-1] if len(self._saved_paths) > 0 else None

    def get_post_training_snapshot_path(self) -> str:
        return self._post_training_snapshot_path


_CHECKPOINT_MANAGER = CheckpointManager()


def set_checkpoint_manager(manager: CheckpointManager) -> None:
    global _CHECKPOINT_MANAGER
    _CHECKPOINT_MANAGER = manager


def save(
    config: PyTextConfig,
    model: Model,
    meta: Optional[CommonMetadata],
    tensorizers: Dict[str, Tensorizer],
    training_state: Optional[TrainingState] = None,
    identifier: Optional[str] = None,
) -> str:
    """
    Save all stateful information of a training task to a specified file-like
    object, will save the original config, model state, metadata,
    training state if training is not completed
    Args:
    identifier (str): used to identify a checkpoint within a training job,
    used as a suffix for save path
    config (PytextConfig): contains all raw parameter/hyper-parameters
    for training task
    model (Model): actual model in training
    training_state (TrainingState): stateful infomation during training
    Returns:
    identifier (str): if identifier is not specified, will save to
    config.save_snapshot_path to be consistent to post-training snapshot;
    if specified, will be used to save checkpoint during training,
    identifier is used to identify checkpoints in the same training
    """
    return _CHECKPOINT_MANAGER.save(
        config, model, meta, tensorizers, training_state, identifier
    )


def load(load_path: str):
    """
    Load task, config and training state from a saved snapshot, will construct
    the task using the saved config then load metadata and model state.
    """
    return _CHECKPOINT_MANAGER.load(load_path)

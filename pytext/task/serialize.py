#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import abc
import io
import logging
import os
from typing import Dict, List, Optional

import torch
from pytext.config import PyTextConfig, config_to_json, pytext_config_from_json
from pytext.data import CommonMetadata
from pytext.data.tensorizers import Tensorizer
from pytext.models import Model
from pytext.trainers.training_state import TrainingState
from pytext.utils.file_io import PathManager
from pytext.utils.usage import log_class_usage


DATA_STATE = "data_state"
CONFIG_JSON = "config_json"
MODEL_STATE = "model_state"
SERIALIZE_VERSION_KEY = "pytext_serialization_version"
TENSORIZERS = "tensorizers"
TRAINING_STATE = "training_state"


LATEST_SERIALIZE_VERSION = 3
LOADER_VERSION_MAP = {}


logger = logging.getLogger(__name__)


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
    return task, config, None


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
    return task, config, None


@register_snapshot_loader(3)
def load_v3(state, overwrite_config=None, rank=0, world_size=1):
    saved_config = pytext_config_from_json(state[CONFIG_JSON])
    if overwrite_config:
        config = overwrite_config
        print(f"Use config from current task")
    else:
        config = saved_config
        print(f"Use config saved in snapshot")

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
        rank=rank,
        world_size=world_size,
    )

    # TODO: T53664090 @stevenliu save & load state_dict() of optimizer and scheduler
    if training_state:
        if training_state.model is None and task.model:
            training_state.model = task.model
        if training_state.optimizer and task.trainer.optimizer:
            """
            https://pytorch.org/tutorials/beginner/saving_loading_models.html
            Unpickling optimizer object from checkpoint could result in a
            different parameter copy from model parameters. Especially in
            mixied precision training, which optimizer param_groups maintains
            master weights copy instead of the model parameters.

            The suggested loading mechanism is

            model = TheModelClass(*args, **kwargs)
            optimizer = TheOptimizerClass(model.parameters(), *args, **kwargs)

            checkpoint = torch.load(PATH)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            """

            optimizer = task.trainer.optimizer
            optimizer.load_state_dict(training_state.optimizer.state_dict())
            training_state.optimizer = optimizer

    return task, config, training_state


def load_checkpoint(state, overwrite_config=None, rank=0, world_size=1):
    print(f"Loaded checkpoint...")
    if SERIALIZE_VERSION_KEY not in state:
        return load_v1(state)
    else:
        return LOADER_VERSION_MAP[state[SERIALIZE_VERSION_KEY]](
            state, overwrite_config, rank, world_size
        )


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


def get_latest_checkpoint_path(dir_path: Optional[str] = None) -> str:
    """
    Get the latest checkpoint path
    args:
        dir_path: the dir to scan for existing checkpoint files. Default: if None,
        the latest checkpoint path saved in momery will be returned
    Returns: checkpoint_path
    """
    if not dir_path:
        return _CHECKPOINT_MANAGER.get_latest_checkpoint_path()

    if PathManager.exists(dir_path):
        checkpoint_indices = [
            int(file_path.split("-")[1])
            for file_path in PathManager.ls(dir_path)
            if file_path.startswith("checkpoint")
        ]
        if checkpoint_indices:
            latest_checkpoint_path = f"{dir_path}/checkpoint-{max(checkpoint_indices)}"
            logger.info(f"find the latest checkpoint: {latest_checkpoint_path}")
            return latest_checkpoint_path
    return None


def get_post_training_snapshot_path() -> str:
    return _CHECKPOINT_MANAGER.get_post_training_snapshot_path()


DELIMITER = "-"


# generate per epoch checkpoint save path
def generate_checkpoint_path(config: PyTextConfig, identifier: str):
    dir_name = os.path.dirname(config.save_snapshot_path)
    return f"{dir_name}/checkpoint{DELIMITER}{identifier}"


class PyTextCheckpointManagerInterface(abc.ABC):
    """
    CheckpointManager is a class abstraction to manage a training job's
    checkpoints/snapshots with different IO and storage.
    """

    @abc.abstractmethod
    def save_checkpoint(self, state, checkpoint_path):
        """
        Serialize and save checkpoint to given path. State is a dictionary that
        represents the all data to be saved.
        Args:
            state: Dictionary containing data to be saved
            checkpoint_path: path of file to save checkpoint
        """
        raise NotImplementedError("Not implemented in interface class")

    @abc.abstractmethod
    def save_snapshot(self, state, snapshot_path):
        """
        Serialize and save post-training model snapshot to given path. State
        is a dictionary that represents the all data to be saved.
        Having a separate method for snapshots enables future optimizations like
        quantization to be applied to snapshots.

        Args:
            state: Dictionary containing data to be saved
            snapshot_path: path of file to save snapshot
        """
        raise NotImplementedError("Not implemented in interface class")

    @abc.abstractmethod
    def load(self, load_path: str):
        """
        Loads a checkpoint/snapshot from disk.
        Args:
            load_path (str): the file path from which to load
        Returns: De-serialized state (dictionary) that was saved
        """
        raise NotImplementedError("Not implemented in interface class")

    @abc.abstractmethod
    def list(self) -> List[str]:
        """
        Return all existing checkpoint paths
        Returns: checkpoint_path_list (List[str]), list elements are in the same
        order of checkpoint saving
        """
        raise NotImplementedError("Not implemented in interface class")

    @abc.abstractmethod
    def get_latest_checkpoint_path(self) -> str:
        """
        Return most recent saved checkpoint path in str
        Returns: checkpoint_path (str)
        """
        raise NotImplementedError("Not implemented in interface class")

    @abc.abstractmethod
    def get_post_training_snapshot_path(self) -> str:
        raise NotImplementedError("Not implemented in interface class")


class CheckpointManager(PyTextCheckpointManagerInterface):
    def __init__(self):
        # keep a list of saved checkpoint path
        self._saved_paths: List[str] = []
        self._post_training_snapshot_path = None
        log_class_usage(__class__)

    def save(self, state, save_path):
        with PathManager.open(save_path, "wb") as f:
            torch.save(state, f)

    def save_checkpoint(self, state, checkpoint_path):
        self.save(state, checkpoint_path)
        self._saved_paths.append(checkpoint_path)

    def save_snapshot(self, state, snapshot_path):
        self.save(state, snapshot_path)
        self._post_training_snapshot_path = snapshot_path

    def load(self, load_path: str):
        if not (load_path and PathManager.isfile(load_path)):
            raise ValueError(f"Invalid snapshot path: {load_path}")
        with PathManager.open(load_path, "rb") as checkpoint_f:
            state = torch.load(checkpoint_f, map_location=lambda storage, loc: storage)
        return state

    def list(self) -> List[str]:
        return self._saved_paths

    def get_latest_checkpoint_path(self) -> str:
        return self._saved_paths[-1] if len(self._saved_paths) > 0 else None

    def get_post_training_snapshot_path(self) -> str:
        return self._post_training_snapshot_path


_CHECKPOINT_MANAGER = CheckpointManager()


def set_checkpoint_manager(manager: PyTextCheckpointManagerInterface) -> None:
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
    saved_path = ""
    if identifier:
        # saving during-training checkpoints
        saved_path = generate_checkpoint_path(config, identifier)
    else:
        # saving post-training snapshot if no identifer given
        saved_path = config.save_snapshot_path
        print(f"Saving pytorch model to: {saved_path}")

    saved_folder = os.path.dirname(saved_path)
    if not PathManager.exists(saved_folder):
        PathManager.mkdirs(saved_folder)
        print(f"created {saved_folder}")

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
        if identifier is not None:
            _CHECKPOINT_MANAGER.save_checkpoint(state, saved_path)
        else:
            _CHECKPOINT_MANAGER.save_snapshot(state, saved_path)

    finally:
        if training_state:
            training_state.model = model_in_training_state
    return saved_path


def load(load_path: str, overwrite_config=None, rank=0, world_size=1):
    """
    Load task, config and training state from a saved snapshot
    by default, it will construct the task using the saved config then load
    metadata and model state.

    if overwrite_task is specified, it will construct the task using
    overwrite_task then load metadata and model state.
    """
    state = _CHECKPOINT_MANAGER.load(load_path)
    return load_checkpoint(state, overwrite_config, rank, world_size)

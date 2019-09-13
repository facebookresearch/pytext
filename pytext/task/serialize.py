#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import io
import os
from typing import List, Optional

import torch
from pytext.config import PyTextConfig, config_to_json, pytext_config_from_json
from pytext.data import CommonMetadata
from pytext.trainers.training_state import TrainingState
from pytext.utils import distributed


DATA_STATE = "data_state"
CONFIG_JSON = "config_json"
MODEL_STATE = "model_state"
OPTIMIZER_STATE = "optimizer_state"
SCHEDULER_STATE = "scheduler_state"
SERIALIZE_VERSION_KEY = "pytext_serialization_version"
TENSORIZERS = "tensorizers"
TRAINING_STATE = "training_state"


LATEST_SERIALIZE_VERSION = 4
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
def load_v3(state, overwrite_config=None):
    saved_config = pytext_config_from_json(state[CONFIG_JSON])
    if overwrite_config:
        config = overwrite_config
        distributed.force_print(f"Use config from current task", flush=True)
    else:
        config = saved_config
        distributed.force_print(f"Use config saved in snapshot", flush=True)

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


@register_snapshot_loader(4)
def load_v4(state, overwrite_config=None):
    saved_config = pytext_config_from_json(state[CONFIG_JSON])
    if overwrite_config:
        config = overwrite_config
        distributed.force_print(f"Use config from current task", flush=True)
    else:
        config = saved_config
        distributed.force_print(f"Use config saved in snapshot", flush=True)

    model_state_dict = state[MODEL_STATE]
    optimizer_state_dict = state[OPTIMIZER_STATE]
    scheduler_state_dict = state[SCHEDULER_STATE]
    training_state = state[TRAINING_STATE]

    # importing in file level generates circular import/dependency failures,
    # that need refator later to fix
    from .task import create_task

    task = create_task(
        config.task,
        metadata=state[DATA_STATE],
        model_state=model_state_dict,
        tensorizers=training_state.tensorizers,
        optimizer_state_dict=optimizer_state_dict,
        scheduler_state_dict=scheduler_state_dict,
    )
    training_state.optimizer = task.trainer.optimizer
    # scheduler is optional in trainer
    if task.trainer.scheduler:
        training_state.scheduler = task.trainer.scheduler

    return task, config, training_state


def load_checkpoint(f: io.IOBase, overwrite_config=None):
    state = torch.load(f, map_location=lambda storage, loc: storage)
    distributed.force_print(f"Loaded checkpoint...", flush=True)
    if SERIALIZE_VERSION_KEY not in state:
        return load_v1(state)
    else:
        return LOADER_VERSION_MAP[state[SERIALIZE_VERSION_KEY]](state, overwrite_config)


def save_checkpoint(
    f: io.IOBase,
    config: PyTextConfig,
    training_state: TrainingState,
    meta: Optional[CommonMetadata] = None,
) -> str:
    # Currently torch.save() has error pickling certain models when not saving
    # by model.state_dict(), thus currently overriding the model in
    # training_state with None, and put back saving
    # also applied to optimizer and scheduler
    # https://github.com/pytorch/pytorch/issues/15116
    model = training_state.model
    optimizer = training_state.optimizer
    scheduler = training_state.scheduler
    scheduler_dict = None
    try:
        # scheduler may not have state_dict()
        scheduler_dict = scheduler.state_dict()
    except AttributeError:
        print(f"{scheduler.__class__()} doesn't have state_dict()")

    # avoid duplicate saving by clearing model, optimizer and scheuler in
    # training_state, they will be loaded with the state_dicts above
    training_state.model = None
    training_state.optimizer = None
    training_state.scheduler = None

    try:
        state = {
            DATA_STATE: meta,
            CONFIG_JSON: config_to_json(PyTextConfig, config),
            MODEL_STATE: model.state_dict(),
            OPTIMIZER_STATE: optimizer.state_dict(),
            SCHEDULER_STATE: scheduler_dict,
            SERIALIZE_VERSION_KEY: LATEST_SERIALIZE_VERSION,
            TRAINING_STATE: training_state,
        }
        torch.save(state, f)
    finally:
        training_state.model = model
        training_state.optimizer = optimizer
        training_state.scheduler = scheduler


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
        training_state: TrainingState,
        identifier: str = None,
        meta: Optional[CommonMetadata] = None,
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
            saved_path = save_checkpoint(checkpoint_f, config, training_state, meta)
            if identifier:
                self._saved_paths.append(saved_path)
            else:
                self._post_training_snapshot_path = saved_path
        return save_path

    def load(self, load_path: str, overwrite_config=None):
        """
        Loads a checkpoint from disk.
        Args:
            load_path (str): the file path to load for checkpoint
        Returns: task (Task), config (PyTextConfig) and training_state (TrainingState)
        """
        if not (load_path and os.path.isfile(load_path)):
            raise ValueError(f"Invalid snapshot path{load_path}")
        distributed.force_print(f"Loading model from {load_path}...", flush=True)
        with open(load_path, "rb") as checkpoint_f:
            return load_checkpoint(checkpoint_f, overwrite_config)

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
    training_state: TrainingState,
    identifier: Optional[str] = None,
    meta: Optional[CommonMetadata] = None,
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
    return _CHECKPOINT_MANAGER.save(config, training_state, identifier, meta)


def load(load_path: str, overwrite_config=None):
    """
    Load task, config and training state from a saved snapshot
    by default, it will construct the task using the saved config then load
    metadata and model state.

    if overwrite_task is specified, it will construct the task using
    overwrite_task then load metadata and model state.
    """
    return _CHECKPOINT_MANAGER.load(load_path, overwrite_config)

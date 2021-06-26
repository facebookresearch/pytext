#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import abc
from typing import List

import torch
from pytext.utils.file_io import PathManager
from pytext.utils.usage import log_class_usage


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


def get_checkpoint_manager() -> CheckpointManager:
    return _CHECKPOINT_MANAGER


def set_checkpoint_manager(manager: PyTextCheckpointManagerInterface) -> None:
    global _CHECKPOINT_MANAGER
    _CHECKPOINT_MANAGER = manager

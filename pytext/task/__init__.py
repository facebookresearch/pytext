#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from .new_task import NewDocumentClassification, NewTask
from .serialize import load, save
from .task import Task, TaskBase, create_task


__all__ = [
    "NewTask",
    "NewDocumentClassification",
    "Task",
    "TaskBase",
    "save",
    "load",
    "create_task",
]

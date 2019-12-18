#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import glob
import importlib
import inspect
import os

from pytext.common.utils import eprint
from pytext.config.component import register_tasks
from pytext.task.disjoint_multitask import DisjointMultitask, NewDisjointMultitask
from pytext.task.new_task import NewTask
from pytext.task.task import Task_Deprecated
from pytext.task.tasks import (
    BertPairRegressionTask,
    DocumentClassificationTask,
    DocumentRegressionTask,
    EnsembleTask,
    IntentSlotTask,
    LMTask,
    MaskedLMTask,
    NewBertClassificationTask,
    NewBertPairClassificationTask,
    PairwiseClassificationTask,
    QueryDocumentPairwiseRankingTask,
    SemanticParsingTask,
    SeqNNTask,
    SquadQATask,
    WordTaggingTask,
)
from pytext.utils.file_io import PathManager


USER_TASKS_DIR = "user_tasks"


def add_include(path):
    """
    Import tasks (and associated components) from the folder name.
    """
    eprint("Including:", path)
    modules = glob.glob(os.path.join(path, "*.py"))
    all = [
        os.path.basename(f)[:-3].replace("/", ".")
        for f in modules
        if PathManager.isfile(f) and not f.endswith("__init__.py")
    ]
    for mod_name in all:
        mod_path = path.replace("/", ".") + "." + mod_name
        eprint("... importing module:", mod_path)
        my_module = importlib.import_module(mod_path)

        for m in inspect.getmembers(my_module, inspect.isclass):
            if m[1].__module__ != mod_path:
                pass
            elif Task_Deprecated in m[1].__bases__ or NewTask in m[1].__bases__:
                eprint("... task:", m[1].__name__)
                register_tasks(m[1])
            else:
                eprint("... importing:", m[1])
                importlib.import_module(mod_path, m[1])


def register_builtin_tasks():
    register_tasks(
        (
            BertPairRegressionTask,
            DisjointMultitask,
            DocumentClassificationTask,
            DocumentRegressionTask,
            EnsembleTask,
            IntentSlotTask,
            LMTask,
            MaskedLMTask,
            NewBertClassificationTask,
            NewBertPairClassificationTask,
            NewDisjointMultitask,
            PairwiseClassificationTask,
            QueryDocumentPairwiseRankingTask,
            SemanticParsingTask,
            SeqNNTask,
            SquadQATask,
            WordTaggingTask,
        )
    )

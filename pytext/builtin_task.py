#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from pytext.config.component import register_tasks
from pytext.task.disjoint_multitask import DisjointMultitask
from pytext.task.tasks import (
    ContextualIntentSlotTask,
    DocClassificationTask,
    EnsembleTask,
    JointTextTask,
    LMTask,
    PairClassification,
    SemanticParsingTask,
    SeqNNTask,
    WordTaggingTask,
)


def register_builtin_tasks():
    register_tasks(
        (
            DocClassificationTask,
            WordTaggingTask,
            JointTextTask,
            LMTask,
            EnsembleTask,
            PairClassification,
            SeqNNTask,
            ContextualIntentSlotTask,
            SemanticParsingTask,
            DisjointMultitask,
        )
    )

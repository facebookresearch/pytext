#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Dict

from pytext.common.constants import BatchContext, Stage
from pytext.config.component import Component, ComponentType, create_component
from pytext.data import BaseBatchSampler, Data, EvalBatchSampler, generator_iterator


class DisjointMultitaskData(Data):
    """
    Wrapper for doing multitask training using multiple data objects.
    Takes a dictionary of data objects, does round robin over their
    iterators using BatchSampler.

    Args:
        config (Config): Configuration object of type DisjointMultitaskData.Config.
        data_dict (Dict[str, Data]): Data objects to do roundrobin over.
        *args (type): Extra arguments to be passed down to sub data handlers.
        **kwargs (type): Extra arguments to be passed down to sub data handlers.

    Attributes:
        data_dict (type): Data handlers to do roundrobin over.

    """

    class Config(Component.Config):
        sampler: BaseBatchSampler.Config = EvalBatchSampler.Config()

    def __init__(self, config: Config, data_dict: Dict[str, Data]) -> None:
        self.data_dict = data_dict
        self.samplers = {
            Stage.TRAIN: create_component(ComponentType.BATCH_SAMPLER, config.sampler),
            Stage.EVAL: EvalBatchSampler(),
            Stage.TEST: EvalBatchSampler(),
        }

    @generator_iterator
    def batches(self, stage: Stage, rank=0, world_size=1, data_source=None):
        all_batches = {
            name: task.batches(stage, rank, world_size)
            for name, task in self.data_dict.items()
        }
        for name, batch in self.samplers[stage].batchify(all_batches):
            batch[BatchContext.TASK_NAME] = name
            yield batch

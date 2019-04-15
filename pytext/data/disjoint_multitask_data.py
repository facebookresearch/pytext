#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Dict

from pytext.common.constants import BatchContext, Stage
from pytext.config.component import create_component
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

    class Config(Data.Config):
        sampler: BaseBatchSampler.Config = EvalBatchSampler.Config()

    def __init__(
        self, config: Config, data_dict: Dict[str, Data], *args, **kwargs
    ) -> None:
        self.data_dict = data_dict
        self.sampler_config = config.sampler

    @generator_iterator
    def batches(self, stage: Stage, rank=0, world_size=1, data_source=None):
        all_batches = {
            name: task.batches(stage, rank, world_size)
            for name, task in self.data_dict.items()
        }
        if stage == Stage.TRAIN:
            sampler = create_component(self.sampler_config, iterators=all_batches)
        else:
            sampler = EvalBatchSampler(all_batches)

        for name, batch in sampler.batchify(all_batches):
            batch[BatchContext.TASK_NAME] = name
            yield batch

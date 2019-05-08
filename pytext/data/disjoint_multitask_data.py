#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Dict, Optional

from pytext.common.constants import BatchContext, Stage
from pytext.config.component import Component, ComponentType, create_component
from pytext.data import BaseBatchSampler, Data, EvalBatchSampler


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
        epoch_size: Optional[int] = None
        sampler: BaseBatchSampler.Config = EvalBatchSampler.Config()
        test_key: Optional[str] = None

    @classmethod
    def from_config(
        cls,
        config: Config,
        data_dict: Dict[str, Data],
        task_key: str = BatchContext.TASK_NAME,
        rank=0,
        world_size=1,
    ):
        samplers = {
            Stage.TRAIN: create_component(ComponentType.BATCH_SAMPLER, config.sampler),
            Stage.EVAL: EvalBatchSampler(),
            Stage.TEST: EvalBatchSampler(),
        }
        return cls(data_dict, samplers, config.epoch_size, config.test_key, task_key)

    def __init__(
        self,
        data_dict: Dict[str, Data],
        samplers: Dict[Stage, BaseBatchSampler],
        epoch_size: Optional[int] = None,
        test_key: str = None,
        task_key: str = BatchContext.TASK_NAME,
    ) -> None:
        # Currently the data object needs to specify `data_source` and `tensorizers`
        # to be used at test time. Set these variables to the values of the `test_key`
        # task.
        test_key = test_key or list(data_dict)[0]
        data_source = data_dict[test_key].data_source
        tensorizers = data_dict[test_key].tensorizers
        # Don't pass tensorizers to __init__, or else initialization will happen again.
        super().__init__(data_source, tensorizers={}, epoch_size=epoch_size)
        self.tensorizers = tensorizers
        self.data_dict = data_dict
        self.samplers = samplers
        self.task_key = task_key

    def _get_batches(self, stage, data_source):
        if not self.batch[stage]:
            all_batches = {
                name: task.batches(stage) for name, task in self.data_dict.items()
            }
            self.batch[stage] = self.samplers[stage].batchify(all_batches)

        for name, batch in self.batch[stage]:
            batch[self.task_key] = name
            yield batch

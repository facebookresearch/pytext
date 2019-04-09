#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from collections.abc import Iterator
from typing import Dict, Optional

from pytext.common.constants import BatchContext, Stage
from pytext.config.component import Component, ComponentType, create_component
from pytext.data import Data, generator_iterator


class BaseBatchSampler(Component):
    __COMPONENT_TYPE__ = ComponentType.BATCH_SAMPLER
    __EXPANSIBLE__ = True

    @classmethod
    def from_config(
        cls, config: Component.Config, iterators: Dict[str, Iterator], **kwargs
    ):
        cls(iterators, **kwargs)

    def __init__(self, iterators: Dict[str, Iterator]):
        self.iterators = iterators


class EvalBatchSampler(BaseBatchSampler):
    """
    Go through all items in all iterators in turn.
    """

    def __iter__(self):
        iterators = {name: iter(iterator) for name, iterator in self.iterators.items()}
        for name, it in iterators.items():
            for item in it:
                yield name, item


class RoundRobinBatchSampler(BaseBatchSampler):
    """
    We take a dictionary of Iterators and do round robin over them in a cycle.
    The below describes the behavior for one epoch, with the example

    Iterator 1: [A, B, C, D],  Iterator 2: [a, b]

    If `iter_to_set_epoch` is set, cycle batches from each iterator until one epoch
    of the target iterator is fulfilled. Iterators with fewer batches than the
    target iterator are repeated, so they never run out.

    iter_to_set_epoch = "Iterator 1"
    Output: [A, a, B, b, C, a, D, b]

    If `iter_to_set_epoch` is None, cycle over batches from each iterator until the
    shortest iterator completes one epoch.

    Output: [A, a, B, b]

    Args:
        iterators (Dict[str, Iterator]): Iterators to do roundrobin over.
        iter_to_set_epoch (Optional[str]): Name of iterator to define epoch size.
          If this is not set, epoch size defaults to the length of
          the shortest iterator.
    Attributes:
        iterators (Dict[str, Iterator]): Iterators to do roundrobin over.
        iter_to_set_epoch (str): Name of iterator to define epoch size.
    """

    __COMPONENT_TYPE__ = ComponentType.BATCH_SAMPLER

    class Config(Component.Config):
        iter_to_set_epoch: str = ""

    @classmethod
    def from_config(cls, config: Config, iterators: Dict[str, Iterator]):
        cls(iterators, config.iter_to_set_epoch)

    def __init__(
        self, iterators: Dict[str, Iterator], iter_to_set_epoch: Optional[str] = None
    ) -> None:
        self.iterators = iterators
        self.iter_state = {
            name: iter(iterator) for name, iterator in self.iterators.items()
        }
        self.iter_to_set_epoch = iter_to_set_epoch

    def __iter__(self):
        while True:
            for name, it in self.iter_state.items():
                try:
                    yield name, next(it)
                except StopIteration:
                    new_iter = iter(self.iterators[name])
                    self.iter_state[name] = new_iter
                    if (not self.iter_to_set_epoch) or name == self.iter_to_set_epoch:
                        self.iter_to_set_epoch = name
                        # end of epoch
                        return
                    else:
                        yield name, next(new_iter)


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
        sampler: BaseBatchSampler.Config = RoundRobinBatchSampler.Config()

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

        for name, batch in sampler:
            batch[BatchContext.TASK_NAME] = name
            yield batch

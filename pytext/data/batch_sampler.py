#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from collections.abc import Iterator
from typing import Dict, Optional

from pytext.config.component import Component, ComponentType


class BaseBatchSampler(Component):
    __COMPONENT_TYPE__ = ComponentType.BATCH_SAMPLER
    __EXPANSIBLE__ = True

    @classmethod
    def from_config(cls, config: Component.Config):
        return cls()

    def __init__(self):
        pass

    def batchify(self, iterators: Dict[str, Iterator]):
        pass


class EvalBatchSampler(BaseBatchSampler):
    """
    This sampler takes in a dictionary of Iterators and returns batches
    associated with each key in the dictionary. It guarentees that we  will see
    each batch associated with each key exactly once in the epoch.

    Example:

        Iterator 1: [A, B, C, D], Iterator 2: [a, b]

        Output: [A, B, C, D, a, b]
    """

    def batchify(self, iterators: Dict[str, Iterator]):
        """
        Loop through each key in the input dict and generate batches from
        the iterator associated with that key.

        Args:
            iterators: Dictionary of iterators
        """
        iter_dict = {name: iter(iterator) for name, iterator in iterators.items()}
        for name, it in iter_dict.items():
            for item in it:
                yield name, item


class RoundRobinBatchSampler(BaseBatchSampler):
    """
    This sampler takes a dictionary of Iterators and returns batches in a round
    robin fashion till a the end of one of the iterators is reached. The end
    is specified by `iter_to_set_epoch`.

    If `iter_to_set_epoch` is set, cycle batches from each iterator until one
    epoch of the target iterator is fulfilled. Iterators with fewer batches
    than the target iterator are repeated, so they never run out.

    If `iter_to_set_epoch` is None, cycle over batches from each iterator until the
    shortest iterator completes one epoch.

    Example:

        Iterator 1: [A, B, C, D],  Iterator 2: [a, b]

        iter_to_set_epoch = "Iterator 1"
        Output: [A, a, B, b, C, a, D, b]

        iter_to_set_epoch = None
        Output: [A, a, B, b]

    Args:
        iter_to_set_epoch (Optional[str]): Name of iterator to define epoch size.
            If this is not set, epoch size defaults to the length of
            the shortest iterator.
    """

    __COMPONENT_TYPE__ = ComponentType.BATCH_SAMPLER

    class Config(Component.Config):
        iter_to_set_epoch: str = ""

    @classmethod
    def from_config(cls, config: Config):
        return cls(config.iter_to_set_epoch)

    def __init__(self, iter_to_set_epoch: Optional[str] = None) -> None:
        self.iter_to_set_epoch = iter_to_set_epoch

    def batchify(self, iterators: Dict[str, Iterator]):
        """
        Loop through each key in the input dict and generate batches from
        the iterator associated with that key until the target iterator reaches
        its end.

        Args:
            iterators: Dictionary of iterators
        """
        iter_dict = {name: iter(iterator) for name, iterator in iterators.items()}
        while True:
            for name, it in iter_dict.items():
                try:
                    yield name, next(it)
                except StopIteration:
                    new_iter = iter(iterators[name])
                    iter_dict[name] = new_iter
                    if (not self.iter_to_set_epoch) or name == self.iter_to_set_epoch:
                        self.iter_to_set_epoch = name
                        # end of epoch
                        return
                    else:
                        yield name, next(new_iter)

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from collections.abc import Iterator
from typing import Dict, Optional

import numpy as np
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


class ProbabalisticBatchSampler(BaseBatchSampler):
    """
    This sampler takes in a dictionary of iterators and returns batches according
    to the specified probabilities in iterator_probabilities. The epoch_size
    argument governs the end of the epoch. If epoch_size = -1, then the epoch
    ends when the first iterator runs out of data. Otherwise we cycle through
    the iterators until epoch_size number of batches have been generated.

    A few notes:
    - The keys in iterator_probabilities should be exactly the same as the
    keys in iterators.
    - The values in iterator_probabilities should sum to 1 since they are
    probabilities.

    Example:

        Iterator A: [A, B, C, D],  Iterator B: [a, b]

        epoch_size = -1, iterator_probabilities = {"A": 0, "B": 1}
        Output = [a, b]

        epoch_size = -1, iterator_probabilities = {"A": 1, "B": 0}
        Output = [A, B, C, D]

        epoch_size = 4, iterator_probabilities = {"A": 0, "B": 1}
        Output = [a, b, a, b]
    """

    __COMPONENT_TYPE__ = ComponentType.BATCH_SAMPLER

    class Config(Component.Config):
        iterator_probabilities: Dict[str, float]
        epoch_size: int = -1

    @classmethod
    def from_config(cls, config: Config):
        return cls(config.iterator_probabilities, config.epoch_size)

    def __init__(
        self, iterator_probabilities: Dict[str, float], epoch_size: int = -1
    ) -> None:
        self.epoch_size = epoch_size
        self.iterator_probabilities = iterator_probabilities

    def batchify(self, iterators: Dict[str, Iterator]):
        assert (
            sum(self.iterator_probabilities.values()) == 1.0
        ), "Specified probabilities don't sum to 1."

        for key in iterators.keys():
            assert (
                key in self.iterator_probabilities
            ), "Probability for iterator related to {} is missing".format(key)

        iter_dict = {name: iter(iterator) for name, iterator in iterators.items()}
        self.num_batches = 0

        while True:

            if self.num_batches > 0 and self.num_batches == self.epoch_size:
                # end of epoch
                return

            # Select a candidate iterator using the uniform distribtion
            key_prob_list = list(self.iterator_probabilities.items())
            candidates = [pair[0] for pair in key_prob_list]
            probabilities = [pair[1] for pair in key_prob_list]

            selected_key = np.random.choice(candidates, 1, p=probabilities).item()
            self.num_batches += 1

            try:
                yield selected_key, next(iter_dict[selected_key])
            except StopIteration:

                # if epoch_size is -1 then this ends the epoch
                if self.epoch_size == -1:
                    return
                else:
                    new_iter = iter(iterators[selected_key])
                    iter_dict[selected_key] = new_iter
                    yield selected_key, next(new_iter)

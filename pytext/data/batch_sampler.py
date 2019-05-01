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


class RandomizedBatchSampler(BaseBatchSampler):
    """
    This sampler takes in a dictionary of iterators and returns batches according
    to the specified probabilities by `unnormalized_iterator_probs`. We cycle through
    the iterators (restarting any that "run out") indefinitely.  Set epoch_size in
    Data.Config.

    Example:

        Iterator A: [A, B, C, D],  Iterator B: [a, b]

        epoch_size = 3, unnormalized_iterator_probs = {"A": 0, "B": 1}
        Epoch 1 = [a, b, a]
        Epoch 2 = [b, a, b]

    Args:
        unnormalized_iterator_probs (Dict[str, float]): Iterator sampling probabilities.
            The keys should be the same as the keys of the underlying iterators, and the
            values will be normalized to sum to 1.
    """

    __COMPONENT_TYPE__ = ComponentType.BATCH_SAMPLER

    class Config(Component.Config):
        unnormalized_iterator_probs: Dict[str, float]

    @classmethod
    def from_config(cls, config: Config):
        return cls(config.unnormalized_iterator_probs)

    def __init__(self, unnormalized_iterator_probs: Dict[str, float]) -> None:
        self.iterator_names = list(unnormalized_iterator_probs)
        self.iterator_probs = np.array(
            [float(unnormalized_iterator_probs[name]) for name in self.iterator_names]
        )
        self.iterator_probs /= self.iterator_probs.sum()
        # Note: we need to make `iter_dict` an instance attribute so that it persists
        # across calls to `batchify()`. This way subsequent epochs will continue from
        # previous states of the iterators (instead of recreating them).
        self.iter_dict = None

    def batchify(self, iterators: Dict[str, Iterator]):
        assert set(iterators) == set(self.iterator_names)
        if self.iter_dict is None:
            self.iter_dict = {
                name: iter(iterator) for name, iterator in iterators.items()
            }
        num_batches = 0

        while True:
            # Select a candidate iterator using the uniform distribtion
            selected_key = np.random.choice(self.iterator_names, p=self.iterator_probs)
            try:
                batch = next(self.iter_dict[selected_key])
            except StopIteration:
                self.iter_dict[selected_key] = iter(iterators[selected_key])
                batch = next(self.iter_dict[selected_key])

            num_batches += 1
            yield selected_key, batch

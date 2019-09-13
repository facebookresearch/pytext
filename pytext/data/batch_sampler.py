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


def select_key_and_batch(
    iterator_names: Dict[str, str],
    iterator_probs: Dict[str, float],
    iter_dict: Dict[str, Iterator],
    iterators: Dict[str, Iterator],
):
    """ Helper function for RandomizedBatchSampler and AlternatingRandomizedBatchSampler
    to select a key from iterator_names using iterator_probs and return a batch
    for the selected key using iter_dict and iterators.
    """
    # Select a candidate iterator using the uniform distribtion
    selected_key = np.random.choice(iterator_names, p=iterator_probs)
    try:
        batch = next(iter_dict[selected_key])
    except StopIteration:
        iter_dict[selected_key] = iter(iterators[selected_key])
        batch = next(iter_dict[selected_key])

    return selected_key, batch


def extract_iterator_properties(input_iterator_probs: Dict[str, float]):
    """ Helper function for RandomizedBatchSampler and AlternatingRandomizedBatchSampler
    to generate iterator properties: iterator_names and iterator_probs.
    """
    iterator_names = list(input_iterator_probs)
    iterator_probs = np.array(
        [float(input_iterator_probs[name]) for name in iterator_names]
    )
    iterator_probs /= iterator_probs.sum()

    return iterator_names, iterator_probs


class RandomizedBatchSampler(BaseBatchSampler):
    """
    This sampler takes in a dictionary of iterators and returns batches according
    to the specified probabilities by `unnormalized_iterator_probs`. We cycle through
    the iterators (restarting any that "run out") indefinitely. Set batches_per_epoch
    in Trainer.Config.

    Example:

        Iterator A: [A, B, C, D],  Iterator B: [a, b]

        batches_per_epoch = 3, unnormalized_iterator_probs = {"A": 0, "B": 1}
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
        self.iterator_names, self.iterator_probs = extract_iterator_properties(
            unnormalized_iterator_probs
        )
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
            selected_key, batch = select_key_and_batch(
                self.iterator_names, self.iterator_probs, self.iter_dict, iterators
            )
            num_batches += 1
            yield selected_key, batch


class AlternatingRandomizedBatchSampler(RandomizedBatchSampler):
    """
    This sampler takes in a dictionary of iterators and returns batches alternating
    between keys and probabilities specified by `unnormalized_iterator_probs` and
    'second_unnormalized_iterator_probs', This is used for example in XLM
    pre-training where we alternate between MLM and TLM batches.
    """

    __COMPONENT_TYPE__ = ComponentType.BATCH_SAMPLER

    class Config(Component.Config):
        unnormalized_iterator_probs: Dict[str, float]
        second_unnormalized_iterator_probs: Dict[str, float]

    @classmethod
    def from_config(cls, config: Config):
        assert (
            len(config.unnormalized_iterator_probs) > 0
            and len(config.second_unnormalized_iterator_probs) > 0
        )
        return cls(
            unnormalized_iterator_probs=config.unnormalized_iterator_probs,
            second_unnormalized_iterator_probs=(
                config.second_unnormalized_iterator_probs
            ),
        )

    def __init__(
        self,
        unnormalized_iterator_probs: Dict[str, float],
        second_unnormalized_iterator_probs: Dict[str, float],
    ) -> None:
        super().__init__(unnormalized_iterator_probs)

        (
            self.second_iterator_names,
            self.second_iterator_probs,
        ) = extract_iterator_properties(second_unnormalized_iterator_probs)
        self.is_secondary_turn = False

    def batchify(self, iterators: Dict[str, Iterator]):
        assert set(iterators) == set(self.iterator_names).union(
            set(self.second_iterator_names)
        )

        if self.iter_dict is None:
            self.iter_dict = {
                name: iter(iterator) for name, iterator in iterators.items()
            }

        while True:
            curr_iter = (
                self.second_iterator_names
                if self.is_secondary_turn
                else self.iterator_names
            )
            curr_probs = (
                self.second_iterator_probs
                if self.is_secondary_turn
                else self.iterator_probs
            )

            selected_key, batch = select_key_and_batch(
                curr_iter, curr_probs, self.iter_dict, iterators
            )

            self.is_secondary_turn = not self.is_secondary_turn

            yield selected_key, batch

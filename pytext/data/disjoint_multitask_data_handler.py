#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import itertools
from collections import OrderedDict
from typing import Dict, Optional, Tuple

import numpy as np
from pytext.common.constants import BatchContext

from .data_handler import BatchIterator, DataHandler


class RoundRobinBatchIterator(BatchIterator):
    """
    We take a dictionary of BatchIterators and do round robin over them in a cycle.
    The below describes the behavior for one epoch, with the example

    Iterator 1: [A, B, C, D],  Iterator 2: [a, b]

    If `upsample` is True:
        If `iter_to_set_epoch` is set, cycle batches from each iterator until one epoch
        of the target iterator is fulfilled. Iterators with fewer batches than the
        target iterator are repeated, so they never run out.

        iter_to_set_epoch = "Iterator 1"
        Output: [A, a, B, b, C, a, D, b]

        If `iter_to_set_epoch` is None, cycle over batches from each iterator until the
        shortest iterator completes one epoch.

        Output: [A, a, B, b]

    If `upsample` is False:
        Iterate over batches from one epoch of each iterator, with the order among
        iterators uniformly shuffled.

        Possible output: [a, A, B, C, b, D]

    Args:
        iterators (Dict[str, BatchIterator]): Iterators to do roundrobin over.
        upsample (bool): If upsample, keep cycling over each iterator in round-robin.
          Iterators with less batches will get more passes.  If False, we do single
          pass over each iterator, in random order. Evaluation will use upsample=False.
          Default True.
        iter_to_set_epoch (Optional[str]): Name of iterator to define epoch size.
          If upsample is True and this is not set, epoch size defaults to the length of
          the shortest iterator. If upsample is False, this argument is not used.
    Attributes:
        iterators (Dict[str, BatchIterator]): Iterators to do roundrobin over.
        upsample (bool): Whether to upsample iterators with fewer batches.
        iter_to_set_epoch (str): Name of iterator to define epoch size.
    """

    def __init__(
        self,
        iterators: Dict[str, BatchIterator],
        upsample: bool = True,
        iter_to_set_epoch: Optional[str] = None,
    ) -> None:
        self.iterators = iterators
        self.upsample = upsample
        self.iter_to_set_epoch = iter_to_set_epoch

    def __iter__(self):
        return iter(self._upsample_iter() if self.upsample else self._shuffle_iter())

    def _upsample_iter(self):
        if self.iter_to_set_epoch:
            batch_per_iter = len(self.iterators[self.iter_to_set_epoch])
        else:
            batch_per_iter = min(len(iterator) for iterator in self.iterators.values())
        total_batches = len(self.iterators) * batch_per_iter
        iterators = {
            name: iter(self.cycle(iterator))
            for name, iterator in self.iterators.items()
        }

        # chain list of tuples, resulting in round robin
        round_robin = itertools.chain.from_iterable(
            # zip list of iterators,
            # return tuples with one element from each iterator
            itertools.zip_longest(
                *[  # turn into iterator of (name, batch) tuples
                    zip(itertools.repeat(name), iterator)
                    for name, iterator in iterators.items()
                ]
            )
        )

        for i, (name, (input, target, context)) in enumerate(round_robin):
            if i >= total_batches:
                # end of epoch
                return
            context[BatchContext.TASK_NAME] = name
            yield input, target, context

    def _shuffle_iter(self):
        indices = []
        iterators = []
        for i, (name, it) in enumerate(self.iterators.items()):
            indices.extend([i] * len(it))
            iterators.append((name, iter(it)))

        indices = np.array(indices)
        np.random.shuffle(indices)

        for i in indices:
            name, iterator = iterators[i]
            input, target, context = next(iterator)
            context[BatchContext.TASK_NAME] = name
            yield input, target, context

    @classmethod
    def cycle(cls, iterator):
        while True:
            for item in iterator:
                yield item


class DisjointMultitaskDataHandler(DataHandler):
    """
    Wrapper for doing multitask training using multiple data handlers.
    Takes a dictionary of data handlers, does round robin over their
    iterators using RoundRobinBatchIterator.

    Args:
        config (Config): Configuration object of type DisjointMultitaskDataHandler.Config.
        data_handlers (Dict[str, DataHandler]): Data handlers to do roundrobin over.
        target_task_name (Optional[str]): Used to select best epoch, and set batch_per_epoch.
        *args (type): Extra arguments to be passed down to sub data handlers.
        **kwargs (type): Extra arguments to be passed down to sub data handlers.

    Attributes:
        data_handlers (type): Data handlers to do roundrobin over.
        target_task_name (type): Used to select best epoch, and set batch_per_epoch.
        upsample (bool): If upsample, keep cycling over each iterator in round-robin.
          Iterators with less batches will get more passes.  If False, we do single
          pass over each iterator, the ones which run out will sit idle.  This is
          used for evaluation.  Default True.

    """

    class Config(DataHandler.Config):
        """Configuration class for `DisjointMultitaskDataHandler`.

        Attributes:
            upsample (bool): If upsample, keep cycling over each iterator in round-robin.
              Iterators with less batches will get more passes.  If False, we do single
              pass over each iterator, the ones which run out will sit idle.  This is
              used for evaluation.  Default True.

        """

        upsample: bool = True

    def __init__(
        self,
        config: Config,
        data_handlers: Dict[str, DataHandler],
        target_task_name: Optional[str] = None,
        *args,
        **kwargs
    ) -> None:
        super(DisjointMultitaskDataHandler, self).__init__(config, None, None, None)
        self.data_handlers = data_handlers
        self.upsample = config.upsample
        self.target_task_name = target_task_name

    def get_train_iter(
        self, rank: int = 0, world_size: int = 1
    ) -> Tuple[BatchIterator, ...]:
        iterators: Dict = OrderedDict(
            (name, data_handler.get_train_iter(rank, world_size))
            for name, data_handler in self.data_handlers.items()
        )
        return RoundRobinBatchIterator(
            iterators, upsample=self.upsample, iter_to_set_epoch=self.target_task_name
        )

    def get_eval_iter(self) -> BatchIterator:
        iterators: Dict = OrderedDict(
            (name, data_handler.get_eval_iter())
            for name, data_handler in self.data_handlers.items()
        )
        return RoundRobinBatchIterator(iterators, upsample=False)

    def get_test_iter(self) -> BatchIterator:
        iterators: Dict = OrderedDict(
            (name, data_handler.get_test_iter())
            for name, data_handler in self.data_handlers.items()
        )
        return RoundRobinBatchIterator(iterators, upsample=False)

    def init_metadata(self):
        # get data sets
        self.metadata = {}
        for name, data_handler in self.data_handlers.items():
            data_handler.init_metadata()
            self.metadata[name] = data_handler.metadata

    def load_metadata(self, metadata):
        self.metadata = metadata
        for name, data_handler in self.data_handlers.items():
            data_handler.load_metadata(metadata[name])

    def metadata_to_save(self):
        metadata = {}
        for name, data_handler in self.data_handlers.items():
            metadata[name] = data_handler.metadata_to_save()
        return metadata

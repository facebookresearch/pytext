#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import itertools
from collections import OrderedDict
from typing import Dict, Optional, Tuple

from pytext.common.constants import BatchContext

from .data_handler import BatchIterator, DataHandler


class RoundRobinBatchIterator(BatchIterator):
    """We take a dictionary of BatchIterators and do round robin over
    them in a cycle.  If epoch_size is specified, each iterator is also
    wrapped in a cycle so that they never run out.  Otherwise, a single
    pass is done over each iterator at each epoch.  Iterators that run
    out are filtered out.  Currently there is no re-shuffling of data,
    data order is the same at each epoch.

    e.g.  Iterator 1: [A, B, C, D],  Iterator 2: [a, b]
    Case 1, epoch size is set:
        Output: [A, a, B, b, C, a, D, b, A, ...]
        Here, tasks with less data are effectively upsampled and data is
        balanced accross tasks.
    Case 2, epoch size not set:
        Output: [A, a, B, b, C, D, A, a, B, b, ...]

    Args:
        iterators (Dict[str, BatchIterator]): Iterators to do roundrobin over.
        epoch_size (Optional[int]): Size of epoch in number of batches.  If not set,
        do a single pass over the training data.

    Attributes:
        iterators (type): Iterators to do roundrobin over.
        epoch_size (type): Size of epoch in number of batches.

    """

    def __init__(
        self, iterators: Dict[str, BatchIterator], epoch_size: Optional[int] = None
    ) -> None:
        self.iterators = iterators
        self.epoch_size = epoch_size or float("inf")

    def __iter__(self):
        iterators = {
            name: iter(
                self.cycle(iterator) if (self.epoch_size < float("inf")) else iterator
            )
            for name, iterator in self.iterators.items()
        }

        round_robin = itertools.filterfalse(  # filter iterators that run out
            lambda x: not bool(x),
            # chain list of tuples, resulting in round robin
            itertools.chain.from_iterable(
                # zip list of iterators,
                # return tuples with one element from each iterator
                itertools.zip_longest(
                    *[  # turn into iterator of (name, batch) tuples
                        zip(itertools.repeat(name), iterator)
                        for name, iterator in iterators.items()
                    ]
                )
            ),
        )

        for i, (name, (input, target, context)) in enumerate(round_robin):
            if i >= self.epoch_size:
                # end of epoch
                return
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
        *args (type): Extra arguments to be passed down to sub data handlers.
        **kwargs (type): Extra arguments to be passed down to sub data handlers.

    Attributes:
        data_handlers (type): Data handlers to do roundrobin over.
        epoch_size (type): Size of epoch in number of batches.
        epoch_size (Optional[int]): Size of epoch in number of batches.  If not set,
        do a single pass over the training data.

    """

    class Config(DataHandler.Config):
        """Configuaration class for `DisjointMultitaskDataHandler`.

        Attributes:
            epoch_size (Optional[int]): Size of epoch in number of batches.  If not set,
            do a single pass over the training data.

        """

        epoch_size: Optional[int] = 10

    def __init__(
        self, config: Config, data_handlers: Dict[str, DataHandler], *args, **kwargs
    ) -> None:
        super(DisjointMultitaskDataHandler, self).__init__(config, None, None, None)
        self.data_handlers = data_handlers
        self.epoch_size = config.epoch_size if config.epoch_size else None

    def get_train_iter(
        self, rank: int = 0, world_size: int = 1
    ) -> Tuple[BatchIterator, ...]:
        iterators: Dict = OrderedDict(
            (name, data_handler.get_train_iter(rank, world_size))
            for name, data_handler in self.data_handlers.items()
        )
        return RoundRobinBatchIterator(iterators, epoch_size=self.epoch_size)

    def get_eval_iter(self) -> BatchIterator:
        iterators: Dict = OrderedDict(
            (name, data_handler.get_eval_iter())
            for name, data_handler in self.data_handlers.items()
        )
        return RoundRobinBatchIterator(iterators)

    def get_test_iter(self) -> BatchIterator:
        iterators: Dict = OrderedDict(
            (name, data_handler.get_test_iter())
            for name, data_handler in self.data_handlers.items()
        )
        return RoundRobinBatchIterator(iterators)

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

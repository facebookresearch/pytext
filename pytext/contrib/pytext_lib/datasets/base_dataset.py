#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import itertools
import math
import random
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from pytext.data.sources.data_source import shard
from pytext.torchscript.utils import long_tensor_2d
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset

from ..transforms import Transform


class BaseDataset(IterableDataset):
    def __init__(
        self,
        iterable: Iterable,
        batch_size: Optional[int] = None,
        is_shuffle: bool = True,
        transforms_dict: Dict[str, List[Transform]] = None,
        batcher=None,
        collate_fn=None,
        chunk_size: int = 1000,
        is_cycle: bool = False,
        length: Optional[int] = None,
        rank: int = 0,
        num_workers: int = 1,
    ):
        self.iterable = itertools.cycle(iterable) if is_cycle else iterable
        if num_workers > 1:
            self.iterable = shard(self.iterable, rank, num_workers)
        self.batch_size = batch_size or batcher.batch_size
        self.batcher = batcher or Batcher(self.batch_size)
        self.is_shuffle = is_shuffle
        self.transforms_dict = transforms_dict or {}
        self.collate_fun = collate_fn or default_collate_fn

        self.chunk_size = chunk_size  # num of batches per chunk
        self.is_cycle = is_cycle
        self.length = length

        self.iterable = ChunkIterator(
            self.iterable, self.chunk_size * self.batch_size, self.length
        )

    def __iter__(self):
        for chunk in iter(self.iterable):
            if self.is_shuffle:
                random.shuffle(chunk)
            transformed_chunk = []
            for row in chunk:
                transformed_row = {}
                for column_name, transforms in self.transforms_dict.items():
                    value = row[column_name]
                    for transform in transforms:
                        value = transform(value)
                    transformed_row[column_name] = value
                transformed_chunk.append(transformed_row)
            for batch in self.batcher.batchify(transformed_chunk):
                yield self.collate_fun(batch)

    def transform(self, transforms_dict):
        self.transforms_dict = transforms_dict
        return self

    def batch(self, batch_size: Optional[int] = None, batcher=None):
        assert batch_size or batcher
        self.batch_size = batch_size or batcher.batch_size
        self.batcher = batcher or Batcher(self.batch_size)
        return self

    def shuffle(self, is_shuffle):
        self.is_shuffle = is_shuffle
        return self

    def collate(self, collate_fun):
        self.collate_fun = collate_fun
        return self

    def worker_init_fn(self, worker_id):
        """TODO: sharding for multi process reading in single node and multi-nodes"""
        raise NotImplementedError()


class ChunkIterator:
    def __init__(self, iterator, chunk_size: int, length: Optional[int]):
        self.iterator = iterator
        self.chunk_size = chunk_size
        self.length = length

    def __iter__(self):
        data = []
        for i, example in enumerate(self.iterator):
            if self.length and i >= self.length:
                break
            data.append(example)
            if len(data) == self.chunk_size:
                yield data
                data = []

        if len(data) > 0:
            yield data
            data = []


class Batcher:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def batchify(self, examples):
        for batch in self._group(examples, self.batch_size):
            yield batch

    def _group(self, examples: List[Any], group_size: int = 1, sort_key=None):
        group = []
        for example in examples:
            group.append(example)
            if len(group) == group_size:
                if sort_key:
                    group.sort(key=sort_key, reverse=True)
                yield group
                group = []
        # the last batch
        if len(group) > 0:
            if sort_key:
                group.sort(key=sort_key, reverse=True)
            yield group


class PoolingBatcher(Batcher):
    def __init__(self, batch_size, pool_num_batches=1000, sort_key=None):
        self.batch_size = batch_size
        self.pool_num_batches = pool_num_batches
        self.sort_key = sort_key

    def batchify(self, examples: List[Any]) -> List[Any]:
        """
        1. Load a pool (`batch_size * pool_num_batches` rows).
        2. Sort rows, if necessary.
        3. Shuffle the order in which the batches are returned, if necessary.
        """
        pool_size = self.batch_size * self.pool_num_batches

        for pool in self._group(examples, pool_size, self.sort_key):
            batch_indices = list(range(math.ceil(len(pool) / self.batch_size)))
            if self.sort_key:
                random.shuffle(batch_indices)
            for batch_index in batch_indices:
                batch = pool[
                    self.batch_size * batch_index : self.batch_size * (batch_index + 1)
                ]
                yield batch


@torch.jit.script
def list_max(inputs: List[int]) -> int:
    max_value = inputs[0]  # fine to throw if empty
    for i in range(len(inputs) - 1):  # don't forget the +1
        max_value = max(max_value, inputs[i + 1])
    return max_value


@torch.jit.script
def pad_2d(
    batch: List[torch.Tensor], seq_lens: List[int], pad_idx: int
) -> torch.Tensor:
    pad_to_length = list_max(seq_lens)
    for i in range(len(batch)):
        sentence = batch[i]
        padding = torch.tensor(
            [pad_idx] * (pad_to_length - len(sentence)), dtype=torch.long
        )
        batch[i] = torch.cat((sentence, padding))
    return torch.stack(batch)


@torch.jit.script
def pad_2d_mask(
    inputs: List[torch.Tensor], pad_value: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad a list to a 2d tensor. Returns a pair of tensors, the padded tensor
    as well as a mask tensor. The mask tensor has the same shape as the padded tensor,
    with a 1 in the position of non-pad values and a 0 in the position of pads."""
    max_len = 0
    for i in inputs:
        max_len = max(max_len, len(i))
    tensor = long_tensor_2d((len(inputs), max_len), pad_value)
    mask = long_tensor_2d((len(inputs), max_len), 0)
    for i in range(len(inputs)):
        for j in range(len(inputs[i])):
            tensor[i][j] = inputs[i][j]
            mask[i][j] = 1
    return tensor, mask


@torch.jit.script
def base_collate_fn(
    batch: List[Dict[str, Dict[str, torch.Tensor]]]
) -> Dict[str, List[torch.Tensor]]:
    dic: Dict[str, List[torch.Tensor]] = {}
    for example in batch:
        for _, transformed_rows in example.items():
            for k, v in transformed_rows.items():
                if len(v.size()) == 0:
                    # convert 0 dim to 1 dim tensor,
                    # as cat() doesn't work for 0-dim tensor
                    v = v.unsqueeze(0)
                if k in dic:
                    dic[k].append(v)
                else:
                    dic[k] = [v]
    return dic


def default_collate_fn(
    batch: List[Dict[str, Dict[str, torch.Tensor]]]
) -> Dict[str, torch.Tensor]:
    dic: Dict[str, List[torch.Tensor]] = base_collate_fn(batch)
    for k, v in dic.items():
        dic[k] = pad_sequence(v).transpose_(0, 1)
    return dic


@torch.jit.script
def docnn_collate_fn(
    batch: List[Dict[str, Dict[str, torch.Tensor]]], pad_idx: int = 1
) -> Dict[str, torch.Tensor]:
    model_inputs: Dict[str, List[torch.Tensor]] = base_collate_fn(batch)
    label_ids = (
        torch.stack(model_inputs["label_ids"])
        if "label_ids" in model_inputs
        else torch.empty(0)
    )
    token_ids: List[torch.Tensor] = base_collate_fn(batch)["token_ids"]
    seq_lens: List[int] = []
    for token_ids_per_sequence in token_ids:
        seq_lens.append(len(token_ids_per_sequence))
    padded_token_ids = pad_2d(token_ids, seq_lens=seq_lens, pad_idx=pad_idx)

    return {"token_ids": padded_token_ids, "label_ids": label_ids}


@torch.jit.script
def roberta_collate_fn(
    batch: List[Dict[str, Dict[str, torch.Tensor]]], pad_idx: int = 1
) -> Dict[str, torch.Tensor]:
    model_inputs: Dict[str, List[torch.Tensor]] = base_collate_fn(batch)
    label_ids = (
        torch.stack(model_inputs["label_ids"])
        if "label_ids" in model_inputs
        else torch.empty(0)
    )
    token_ids: List[torch.Tensor] = model_inputs["token_ids"]
    segment_labels: List[torch.Tensor] = []
    positions: List[torch.Tensor] = []
    seq_lens: List[int] = []
    for token_ids_per_sequence in token_ids:
        seq_len = len(token_ids_per_sequence)
        seq_lens.append(seq_len)
        segment_labels.append(torch.tensor([0] * seq_len, dtype=torch.long))
        positions.append(
            torch.tensor([i for i in range(seq_len)], dtype=torch.long)  # noqa
        )

    padded_token_ids, pad_mask = pad_2d_mask(token_ids, pad_value=pad_idx)
    padded_segment_labels = pad_2d(segment_labels, seq_lens=seq_lens, pad_idx=0)
    padded_positions = pad_2d(positions, seq_lens=seq_lens, pad_idx=0)

    return {
        "token_ids": padded_token_ids,
        "pad_mask": pad_mask,
        "segment_labels": padded_segment_labels,
        "positions": padded_positions,
        "label_ids": label_ids,
    }

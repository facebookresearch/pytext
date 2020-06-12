#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Iterable

from pytext.common.constants import Stage

from .data import BatchData, PoolingBatcher
from .sources import RawExample


class TokenBatcher(PoolingBatcher):
    """Batcher that operates on number of tokens rather than number of instances."""

    class Config(PoolingBatcher.Config):
        bsz_mult: int = 1
        max_tokens: int = 16

        def num_tokens_fn(self, item):
            return len(item.split())

    @classmethod
    def from_config(cls, config: Config):
        return cls(
            config.train_batch_size,
            config.eval_batch_size,
            config.test_batch_size,
            config.pool_num_batches,
            config.num_shuffled_pools,
            config.bsz_mult,
            config.max_tokens,
            config.num_tokens_fn,
        )

    def __init__(
        self,
        train_batch_size=Config.train_batch_size,
        eval_batch_size=Config.eval_batch_size,
        test_batch_size=Config.test_batch_size,
        pool_num_batches=Config.pool_num_batches,
        num_shuffled_pools=Config.num_shuffled_pools,
        bsz_mult=Config.bsz_mult,
        max_tokens=Config.max_tokens,
        num_tokens_fn=Config.num_tokens_fn,
    ):
        super().__init__(
            train_batch_size,
            eval_batch_size,
            test_batch_size,
            pool_num_batches,
            num_shuffled_pools,
        )
        self.bsz_mult = bsz_mult
        self.max_tokens = max_tokens
        self.num_tokens_fn = num_tokens_fn

    def _is_batch_full(self, batch, num_tokens: int, max_tokens: int):
        if len(batch) == 0:
            return 0
        if max_tokens > 0 and num_tokens > max_tokens:
            return 1
        return 0

    def batchify(
        self, iterable: Iterable[RawExample], sort_key=None, stage=Stage.TRAIN
    ):

        max_tokens = self.max_tokens
        bsz_mult = self.bsz_mult
        sample_len = 0
        sample_lens = []
        batch = []
        batches = []

        for item in super().batchify(iterable=iterable, sort_key=sort_key, stage=stage):
            num_tokens = self.num_tokens_fn(item[0][0]["source_sequence"])
            sample_lens.append(num_tokens)
            sample_len = max(sample_len, num_tokens)

            assert max_tokens <= 0 or sample_len <= max_tokens, (
                "sentence at index {} of size {} exceeds max_tokens "
                "limit of {}!".format(item, sample_len, max_tokens)
            )
            num_tokens = (len(batch) + 1) * sample_len
            if self._is_batch_full(batch, num_tokens, max_tokens):
                mod_len = max(
                    bsz_mult * (len(batch) // bsz_mult), len(batch) % bsz_mult
                )
                batches.append(self.combine_batch_data(batch[:mod_len]))
                batch = batch[mod_len:]
                sample_lens = sample_lens[mod_len:]
                sample_len = max(sample_lens) if len(sample_lens) > 0 else 0
            batch.append(item)
        if len(batch) > 0:
            batches.append(self.combine_batch_data(batch))
        for batch in batches:
            yield batch

    def combine_batch_data(self, batch) -> BatchData:
        raw_batch = []
        numberized_batch = {}
        for batch_data in batch:
            raw_batch.extend(batch_data[0])
            numberized_batch.update(batch_data[1])
        return BatchData(raw_batch, numberized_batch)

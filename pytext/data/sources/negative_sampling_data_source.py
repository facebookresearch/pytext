#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Dict, Type

from .data_source import (
    RootDataSource,
    SafeFileWrapper,
    ShardedDataSource,
    generator_property,
)
from .tsv import BlockShardedTSV, TSVDataSource


class NegativeSamplingDataSource(RootDataSource, ShardedDataSource):
    """DataSource which returns two independent rows from the same csv file.
    Field names are duplicated and appended with row id.  E.g. if the original
    source has ['text', 'label'], we'll get ['text1', 'text2', 'label1', 'label2']"""

    class Config(TSVDataSource.Config):
        pass

    @classmethod
    def make_tsv(cls, fname, config, rank, world_size, block_id, num_blocks=2):
        if not fname:
            return None
        field_names = [name + str(block_id + 1) for name in config.field_names]

        return BlockShardedTSV(
            SafeFileWrapper(fname),
            field_names=field_names,
            delimiter=config.delimiter,
            block_id=rank + block_id * world_size,
            num_blocks=world_size * num_blocks,
        )

    @classmethod
    def from_config(
        cls,
        config: TSVDataSource.Config,
        schema: Dict[str, Type],
        rank: int = 0,
        world_size: int = 1,
    ):
        args = config._asdict()
        train_filename = args.pop("train_filename")
        train1 = cls.make_tsv(train_filename, config, rank, world_size, 0)
        train2 = cls.make_tsv(train_filename, config, rank, world_size, 1)
        train_unsharded1 = cls.make_tsv(train_filename, config, 0, 1, 0)
        train_unsharded2 = cls.make_tsv(train_filename, config, 0, 1, 1)

        test_filename = args.pop("test_filename")
        test1 = cls.make_tsv(test_filename, config, 0, 1, 0)
        test2 = cls.make_tsv(test_filename, config, 0, 1, 1)

        eval_filename = args.pop("eval_filename")
        eval1 = cls.make_tsv(eval_filename, config, 0, 1, 0)
        eval2 = cls.make_tsv(eval_filename, config, 0, 1, 1)

        return cls(
            schema,
            train1,
            train2,
            train_unsharded1,
            train_unsharded2,
            test1,
            test2,
            eval1,
            eval2,
        )

    def __init__(
        self,
        schema,
        train1,
        train2,
        train_unsharded1,
        train_unsharded2,
        test1,
        test2,
        eval1,
        eval2,
    ):
        # calls init of RootDataSource
        super().__init__(schema=schema)
        # weird python syntax to call init of ShardedDataSource
        super(RootDataSource, self).__init__(schema=schema)
        self.train1 = train1
        self.train2 = train2
        self.test1 = test1
        self.test2 = test2
        self.eval1 = eval1
        self.eval2 = eval2
        self.train_unsharded1 = train_unsharded1
        self.train_unsharded2 = train_unsharded2

    @generator_property
    def train_unsharded(self):
        return iter(self._train_unsharded)

    def raw_data_generator(self, source1, source2):
        if not source1:
            return
        for dict1, dict2 in zip(iter(source1), iter(source2)):
            dict1.update(dict2)
            yield dict1

    def raw_train_data_generator(self):
        return self.raw_data_generator(self.train1, self.train2)

    def raw_test_data_generator(self):
        return self.raw_data_generator(self.test1, self.test2)

    def raw_eval_data_generator(self):
        return self.raw_data_generator(self.eval1, self.eval2)

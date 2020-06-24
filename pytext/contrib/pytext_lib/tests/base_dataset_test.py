#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import unittest

import torch
from pytext.contrib.pytext_lib.datasets import BaseDataset, PoolingBatcher
from pytext.contrib.pytext_lib.transforms import (
    VocabTransform,
    WhitespaceTokenizerTransform,
)
from pytext.data.utils import Vocabulary


class TestBaseDataset(unittest.TestCase):
    def setUp(self):
        self.input_iterator = [
            {"text": "hello world"},
            {"text": "feeling lucky today"},
            {"text": "hello"},
            {"text": "lucky world"},
            {"text": "today world"},
        ]
        self.vocab = Vocabulary(["hello", "world", "feeling", "lucky", "today"])

    def test_base_dataset(self):
        transform_dict = {
            "text": [WhitespaceTokenizerTransform(), VocabTransform(self.vocab)]
        }
        ds = BaseDataset(
            iterable=self.input_iterator,
            batch_size=2,
            is_shuffle=False,
            transforms_dict=transform_dict,
        )
        batches = list(ds)
        assert len(batches) == 3
        assert torch.all(
            batches[0]["token_ids"].eq(torch.tensor([[0, 1, 0], [2, 3, 4]]))
        )
        assert torch.all(batches[1]["token_ids"].eq(torch.tensor([[0, 0], [3, 1]])))
        assert torch.all(batches[2]["token_ids"].eq(torch.tensor([[4, 1]])))

    def test_ds_with_pooling_batcher(self):
        transform_dict = {
            "text": [WhitespaceTokenizerTransform(), VocabTransform(self.vocab)]
        }
        ds = BaseDataset(
            iterable=self.input_iterator,
            batch_size=2,
            is_shuffle=False,
            transforms_dict=transform_dict,
        )
        ds.batch(batcher=PoolingBatcher(2))
        batches = list(ds)
        assert len(batches) == 3
        # in [0, 1, 0], the trailing 0 is padding index
        assert torch.all(
            batches[0]["token_ids"].eq(torch.tensor([[0, 1, 0], [2, 3, 4]]))
        )

    def test_multi_workers_reading(self):
        transform_dict = {
            "text": [WhitespaceTokenizerTransform(), VocabTransform(self.vocab)]
        }
        ds0 = BaseDataset(
            iterable=self.input_iterator,
            batch_size=1,
            is_shuffle=False,
            transforms_dict=transform_dict,
            rank=0,
            num_workers=2,
        )
        ds1 = BaseDataset(
            iterable=self.input_iterator,
            batch_size=1,
            is_shuffle=False,
            transforms_dict=transform_dict,
            rank=1,
            num_workers=2,
        )
        batches0 = list(ds0)
        batches1 = list(ds1)
        # expect ds0 and ds1 to read different partitions of the data
        # the last (len(input_iterator) % num_workers) rows of the data
        # will be discarded because distributed training needs to be in sync
        assert len(batches0) == len(batches1) == 2
        assert torch.all(batches0[0]["token_ids"].eq(torch.tensor([[0, 1]])))
        assert torch.all(batches1[0]["token_ids"].eq(torch.tensor([[2, 3, 4]])))
        assert torch.all(batches0[1]["token_ids"].eq(torch.tensor([[0]])))
        assert torch.all(batches1[1]["token_ids"].eq(torch.tensor([[3, 1]])))

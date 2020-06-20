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
        self.input_iterator = [{"text": "hello world"}, {"text": "feeling lucky today"}]
        self.vocab = Vocabulary(["hello", "world", "feeling", "lucky", "today"])

    def test_base_dataset(self):
        transform_dict = {
            "text": [WhitespaceTokenizerTransform(), VocabTransform(self.vocab)]
        }
        ds = BaseDataset(
            iterable=self.input_iterator,
            batch_size=1,
            is_shuffle=False,
            transforms_dict=transform_dict,
        )
        batches = list(ds)
        assert len(batches) == 2
        assert torch.all(batches[0]["token_ids"].eq(torch.tensor([[0, 1]])))
        assert torch.all(batches[1]["token_ids"].eq(torch.tensor([[2, 3, 4]])))

    def test_ds_with_pooling_batcher(self):
        transform_dict = {
            "text": [WhitespaceTokenizerTransform(), VocabTransform(self.vocab)]
        }
        ds = BaseDataset(
            iterable=self.input_iterator,
            batch_size=1,
            is_shuffle=False,
            transforms_dict=transform_dict,
        )
        ds.batch(batcher=PoolingBatcher(2))
        batches = list(ds)
        assert len(batches) == 1
        # in [0, 1, 0], the trailing 0 is padding index
        assert torch.all(
            batches[0]["token_ids"].eq(torch.tensor([[0, 1, 0], [2, 3, 4]]))
        )

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

from pytext.common.constants import Stage
from pytext.data import Data, RawBatcher, types
from pytext.data.sources.data_source import SafeFileWrapper
from pytext.data.sources.tsv import TSVDataSource
from pytext.data.tensorizers import LabelTensorizer, WordTensorizer
from pytext.utils.test_utils import import_tests_module


tests_module = import_tests_module()


class DataTest(unittest.TestCase):
    def setUp(self):
        self.data_source = TSVDataSource(
            SafeFileWrapper(tests_module.test_file("train_dense_features_tiny.tsv")),
            SafeFileWrapper(tests_module.test_file("test_dense_features_tiny.tsv")),
            eval_file=None,
            field_names=["label", "slots", "text", "dense"],
            schema={"text": types.Text, "label": types.Label},
        )

        self.tensorizers = {
            "tokens": WordTensorizer(column="text"),
            "labels": LabelTensorizer(column="label", allow_unknown=True),
        }

    def test_create_data_no_batcher_provided(self):
        data = Data(self.data_source, self.tensorizers)
        batches = list(data.batches(Stage.TRAIN))
        # We should have made at least one non-empty batch
        self.assertTrue(batches)
        batch, tensors = next(iter(batches))
        self.assertTrue(batch)
        self.assertTrue(tensors)

    def test_create_batches(self):
        data = Data(self.data_source, self.tensorizers, RawBatcher(batch_size=16))
        batches = list(data.batches(Stage.TRAIN))
        self.assertEqual(1, len(batches))
        batch, batch_tensors = next(iter(batches))
        self.assertEqual(set(self.tensorizers), set(batch_tensors))
        tokens, seq_lens = batch_tensors["tokens"]
        self.assertEqual((10,), seq_lens.size())
        self.assertEqual((10,), batch_tensors["labels"].size())
        self.assertEqual(10, len(batch))
        example = next(iter(batch))
        self.assertEqual({"text", "label"}, set(example))

    def test_create_batches_different_tensorizers(self):
        tensorizers = {"tokens": WordTensorizer(column="text")}
        data = Data(self.data_source, tensorizers, RawBatcher(batch_size=16))
        batches = list(data.batches(Stage.TRAIN))
        self.assertEqual(1, len(batches))
        batch, batch_tensors = next(iter(batches))
        self.assertEqual({"tokens"}, set(batch_tensors))
        tokens, seq_lens = batch_tensors["tokens"]
        self.assertEqual((10,), seq_lens.size())
        self.assertEqual(10, len(batch))
        example = next(iter(batch))
        self.assertEqual({"text", "label"}, set(example))

    def test_data_initializes_tensorsizers(self):
        tensorizers = {
            "tokens": WordTensorizer(column="text"),
            "labels": LabelTensorizer(column="label"),
        }
        with self.assertRaises(AttributeError):
            # verify WordTensorizer isn't in an initialized state yet
            tensorizers["tokens"].vocab
        Data(self.data_source, tensorizers)
        # Tensorizers should have been initialized
        self.assertEqual(49, len(tensorizers["tokens"].vocab))
        self.assertEqual(7, len(tensorizers["labels"].labels))

    def test_data_iterate_multiple_times(self):
        data = Data(self.data_source, self.tensorizers)
        batches = data.batches(Stage.TRAIN)
        data1 = list(batches)
        data2 = list(batches)
        # We should have made at least one non-empty batch
        self.assertTrue(data1)
        self.assertTrue(data2)
        batch1, _ = data1[0]
        batch2, _ = data2[0]
        # pytorch tensors don't have equals comparisons, so comparing the tensor
        # dicts is non-trivial, but they should also be equal
        self.assertEqual(batch1, batch2)


class RawBatcherTest(unittest.TestCase):
    def test_raw_batcher(self):
        data = range(10)
        batcher = RawBatcher(batch_size=3)
        self.assertEqual(
            [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]], list(batcher.batchify(data))
        )

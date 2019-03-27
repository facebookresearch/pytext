#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

from pytext.common.constants import Stage
from pytext.data import Batcher, Data, PoolingBatcher
from pytext.data.sources.data_source import SafeFileWrapper
from pytext.data.sources.tsv import TSVDataSource
from pytext.data.tensorizers import LabelTensorizer, TokenTensorizer
from pytext.utils.test import import_tests_module


tests_module = import_tests_module()


class DataTest(unittest.TestCase):
    def setUp(self):
        self.data_source = TSVDataSource(
            SafeFileWrapper(tests_module.test_file("train_dense_features_tiny.tsv")),
            SafeFileWrapper(tests_module.test_file("test_dense_features_tiny.tsv")),
            eval_file=None,
            field_names=["label", "slots", "text", "dense"],
            schema={"text": str, "label": str},
        )

        self.tensorizers = {
            "tokens": TokenTensorizer(text_column="text"),
            "labels": LabelTensorizer(label_column="label", allow_unknown=True),
        }

    def test_create_data_no_batcher_provided(self):
        data = Data(self.data_source, self.tensorizers)
        batches = list(data.batches(Stage.TRAIN))
        # We should have made at least one non-empty batch
        self.assertTrue(batches)
        batch = next(iter(batches))
        self.assertTrue(batch)

    def test_create_batches(self):
        data = Data(self.data_source, self.tensorizers, Batcher(train_batch_size=16))
        batches = list(data.batches(Stage.TRAIN))
        self.assertEqual(1, len(batches))
        batch = next(iter(batches))
        self.assertEqual(set(self.tensorizers), set(batch))
        tokens, seq_lens = batch["tokens"]
        self.assertEqual((10,), seq_lens.size())
        self.assertEqual((10,), batch["labels"].size())
        self.assertEqual({"tokens", "labels"}, set(batch))
        self.assertEqual(10, len(tokens))

    def test_create_batches_different_tensorizers(self):
        tensorizers = {"tokens": TokenTensorizer(text_column="text")}
        data = Data(self.data_source, tensorizers, Batcher(train_batch_size=16))
        batches = list(data.batches(Stage.TRAIN))
        self.assertEqual(1, len(batches))
        batch = next(iter(batches))
        self.assertEqual({"tokens"}, set(batch))
        tokens, seq_lens = batch["tokens"]
        self.assertEqual((10,), seq_lens.size())
        self.assertEqual(10, len(tokens))

    def test_data_initializes_tensorsizers(self):
        tensorizers = {
            "tokens": TokenTensorizer(text_column="text"),
            "labels": LabelTensorizer(label_column="label"),
        }
        # verify TokenTensorizer isn't in an initialized state yet
        assert tensorizers["tokens"].vocab is None
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

    def test_sort(self):
        data = Data(
            self.data_source,
            self.tensorizers,
            Batcher(train_batch_size=16),
            sort_key="tokens",
        )
        batches = list(data.batches(Stage.TRAIN))
        batch = next(iter(batches))
        _, seq_lens = batch["tokens"]
        seq_lens = seq_lens.tolist()
        for i in range(len(seq_lens) - 1):
            self.assertTrue(seq_lens[i] >= seq_lens[i + 1])
        # make sure labels are also in the same order of sorted tokens
        self.assertEqual(
            self.tensorizers["labels"].labels[batch["labels"][1]],
            "reminder/set_reminder",
        )
        self.assertEqual(
            self.tensorizers["labels"].labels[batch["labels"][8]], "alarm/snooze_alarm"
        )


class BatcherTest(unittest.TestCase):
    def test_batcher(self):
        data = [{"a": i, "b": 10 + i, "c": 20 + i} for i in range(10)]
        batcher = Batcher(train_batch_size=3)
        batches = list(batcher.batchify(data))
        self.assertEqual(len(batches), 4)
        self.assertEqual(batches[1]["a"], [3, 4, 5])
        self.assertEqual(batches[3]["b"], [19])

    def test_pooling_batcher(self):
        data = [{"a": i, "b": 10 + i, "c": 20 + i} for i in range(10)]
        batcher = PoolingBatcher(train_batch_size=3, pool_num_batches=2)
        batches = list(batcher.batchify(data, sort_key=lambda x: x["a"]))

        self.assertEqual(len(batches), 4)
        a_vals = {a for batch in batches for a in batch["a"]}
        self.assertSetEqual(a_vals, set(range(10)))
        for batch in batches[:2]:
            self.assertGreater(batch["a"][0], batch["a"][-1])
            for a in batch["a"]:
                self.assertLess(a, 6)
        for batch in batches[2:]:
            for a in batch["a"]:
                self.assertGreaterEqual(a, 6)

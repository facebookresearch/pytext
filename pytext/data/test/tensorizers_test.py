#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

import torch
from pytext.data import types
from pytext.data.sources.data_source import SafeFileWrapper
from pytext.data.sources.tsv import TSVDataSource
from pytext.data.tensorizers import (
    CharacterTensorizer,
    LabelTensorizer,
    WordTensorizer,
    initialize_tensorizers,
)
from pytext.utils.test_utils import import_tests_module


tests_module = import_tests_module()


class TensorizersTest(unittest.TestCase):
    def setUp(self):
        self.data = TSVDataSource(
            SafeFileWrapper(tests_module.test_file("train_dense_features_tiny.tsv")),
            SafeFileWrapper(tests_module.test_file("test_dense_features_tiny.tsv")),
            eval_file=None,
            field_names=["label", "slots", "text", "dense"],
            schema={"text": types.Text, "label": types.Label},
        )

    def test_initialize_tensorizers(self):
        tensorizers = {
            "tokens": WordTensorizer(column="text"),
            "labels": LabelTensorizer(column="label"),
            "chars": CharacterTensorizer(column="text"),
        }
        initialize_tensorizers(tensorizers, self.data.train)
        self.assertEqual(49, len(tensorizers["tokens"].vocab))
        self.assertEqual(7, len(tensorizers["labels"].labels))

    def test_initialize_word_tensorizer(self):
        tensorizer = WordTensorizer(column="text")
        init = tensorizer.initialize()
        init.send(None)  # kick
        for row in self.data.train:
            init.send(row)
        init.close()
        self.assertEqual(49, len(tensorizer.vocab))

    def test_create_word_tensors(self):
        tensorizer = WordTensorizer(column="text")
        init = tensorizer.initialize()
        init.send(None)  # kick
        for row in self.data.train:
            init.send(row)
        init.close()

        batch = [
            {"text": types.Text("I want some coffee")},
            {"text": types.Text("Turn it up")},
        ]

        tokens, seq_lens = tensorizer.create_training_tensors(batch)
        self.assertIsInstance(tokens, torch.LongTensor)
        self.assertIsInstance(seq_lens, torch.LongTensor)
        self.assertEqual((2, 4), tokens.size())
        self.assertEqual((2,), seq_lens.size())
        self.assertEqual([[24, 0, 0, 0], [13, 47, 9, 1]], tokens.tolist())
        self.assertEqual([4, 3], seq_lens.tolist())

    def test_create_character_tensors(self):
        tensorizer = CharacterTensorizer(column="text")
        # not initializing because initializing is a no-op for CharacterTensorizer

        s1 = "I want some coffee"
        s2 = "Turn it up"

        ld = len(s1) - len(s2)

        batch = [{"text": types.Text(s1)}, {"text": types.Text(s2)}]
        expected = [
            [ord(c) for c in s1] + [0] * (max(-ld, 0)),
            [ord(c) for c in s2] + [0] * (max(ld, 0)),
        ]

        chars, seq_lens = tensorizer.create_training_tensors(batch)
        self.assertIsInstance(chars, torch.LongTensor)
        self.assertIsInstance(seq_lens, torch.LongTensor)
        self.assertEqual((2, max(len(s1), len(s2))), chars.size())
        self.assertEqual((2,), seq_lens.size())
        self.assertEqual(expected, chars.tolist())
        self.assertEqual([len(s1), len(s2)], seq_lens.tolist())

    def test_initialize_label_tensorizer(self):
        tensorizer = LabelTensorizer(column="label")
        init = tensorizer.initialize()
        init.send(None)  # kick
        for row in self.data.train:
            init.send(row)
        init.close()
        self.assertEqual(7, len(tensorizer.labels))

    def test_create_label_tensors(self):
        tensorizer = LabelTensorizer(column="label")
        init = tensorizer.initialize()
        init.send(None)  # kick
        for row in self.data.train:
            init.send(row)
        init.close()

        batch = [
            {"label": types.Label("weather/find")},
            {"label": types.Label("alarm/set_alarm")},
        ]

        tensor = tensorizer.create_training_tensors(batch)
        self.assertEqual((2,), tensor.size())
        self.assertEqual([6, 1], tensor.tolist())

    def test_create_label_tensors_fails_with_unknown_label(self):
        tensorizer = LabelTensorizer(column="label")
        init = tensorizer.initialize()
        init.send(None)  # kick
        for row in self.data.train:
            init.send(row)
        init.close()

        batch = [
            {"label": types.Label("non/existent")},
            {"label": types.Label("alarm/set_alarm")},
        ]

        with self.assertRaises(Exception):
            tensorizer.create_training_tensors(batch)

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

import torch
from pytext.data.sources.data_source import SafeFileWrapper
from pytext.data.sources.tsv import TSVDataSource
from pytext.data.tensorizers import (
    ByteTensorizer,
    CharacterTokenTensorizer,
    FloatListTensorizer,
    LabelTensorizer,
    TokenTensorizer,
    initialize_tensorizers,
)
from pytext.utils.test import import_tests_module


tests_module = import_tests_module()


class TensorizersTest(unittest.TestCase):
    def setUp(self):
        self.data = TSVDataSource(
            SafeFileWrapper(tests_module.test_file("train_dense_features_tiny.tsv")),
            SafeFileWrapper(tests_module.test_file("test_dense_features_tiny.tsv")),
            eval_file=None,
            field_names=["label", "slots", "text", "dense"],
            schema={"text": str, "label": str},
        )

    def test_initialize_tensorizers(self):
        tensorizers = {
            "tokens": TokenTensorizer(text_column="text"),
            "labels": LabelTensorizer(label_column="label"),
            "chars": ByteTensorizer(text_column="text"),
        }
        initialize_tensorizers(tensorizers, self.data.train)
        self.assertEqual(49, len(tensorizers["tokens"].vocab))
        self.assertEqual(7, len(tensorizers["labels"].labels))

    def test_initialize_word_tensorizer(self):
        tensorizer = TokenTensorizer(text_column="text")
        init = tensorizer.initialize()
        init.send(None)  # kick
        for row in self.data.train:
            init.send(row)
        init.close()
        self.assertEqual(49, len(tensorizer.vocab))

    def test_create_word_tensors(self):
        tensorizer = TokenTensorizer(text_column="text")
        init = tensorizer.initialize()
        init.send(None)  # kick
        for row in self.data.train:
            init.send(row)
        init.close()

        rows = [{"text": "I want some coffee"}, {"text": "Turn it up"}]
        tensors = (tensorizer.numberize(row) for row in rows)
        tokens, seq_len = next(tensors)
        self.assertEqual([24, 0, 0, 0], tokens)
        self.assertEqual(4, seq_len)

        tokens, seq_len = next(tensors)
        self.assertEqual([13, 47, 9], tokens)
        self.assertEqual(3, seq_len)

    def test_create_byte_tensors(self):
        tensorizer = ByteTensorizer(text_column="text", lower=False)
        # not initializing because initializing is a no-op for ByteTensorizer

        s1 = "I want some coffee"
        s2 = "Turn it up"
        s3 = "我不会说中文"
        rows = [{"text": s1}, {"text": s2}, {"text": s3}]
        expected = [list(s1.encode()), list(s2.encode()), list(s3.encode())]

        tensors = [tensorizer.numberize(row) for row in rows]
        self.assertEqual([(bytes, len(bytes)) for bytes in expected], tensors)

    def test_create_word_character_tensors(self):
        tensorizer = CharacterTokenTensorizer(text_column="text")
        # not initializing because initializing is a no-op for ByteTensorizer

        s1 = "I want some coffee"
        s2 = "Turn it up"

        def ords(word, pad_to):
            return [ord(c) for c in word] + [0] * (pad_to - len(word))

        batch = [{"text": s1}, {"text": s2}]
        # Note that the tokenizer lowercases here
        expected = [
            [ords("i", 6), ords("want", 6), ords("some", 6), ords("coffee", 6)],
            [ords("turn", 6), ords("it", 6), ords("up", 6), ords("", 6)],
        ]

        expected_lens = [[1, 4, 4, 6], [4, 2, 2, 0]]

        chars, seq_lens = tensorizer.tensorize(
            tensorizer.numberize(row) for row in batch
        )
        self.assertIsInstance(chars, torch.LongTensor)
        self.assertIsInstance(seq_lens, torch.LongTensor)
        self.assertEqual((2, 4, 6), chars.size())
        self.assertEqual((2, 4), seq_lens.size())
        self.assertEqual(expected, chars.tolist())
        self.assertEqual(expected_lens, seq_lens.tolist())

    def test_initialize_label_tensorizer(self):
        tensorizer = LabelTensorizer(label_column="label")
        init = tensorizer.initialize()
        init.send(None)  # kick
        for row in self.data.train:
            init.send(row)
        init.close()
        self.assertEqual(7, len(tensorizer.labels))

    def test_create_label_tensors(self):
        tensorizer = LabelTensorizer(label_column="label")
        init = tensorizer.initialize()
        init.send(None)  # kick
        for row in self.data.train:
            init.send(row)
        init.close()

        rows = [
            {"label": "weather/find"},
            {"label": "alarm/set_alarm"},
            {"label": "non/existent"},
        ]

        tensors = (tensorizer.numberize(row) for row in rows)
        tensor = next(tensors)
        self.assertEqual(6, tensor)
        tensor = next(tensors)
        self.assertEqual(1, tensor)
        with self.assertRaises(Exception):
            tensor = next(tensors)

    def test_create_float_list_tensor(self):
        tensorizer = FloatListTensorizer(column="dense")
        rows = [
            {"dense": "[0.1,0.2]"},  # comma
            {"dense": "[0.1, 0.2]"},  # comma with single space
            {"dense": "[0.1,  0.2]"},  # comma with multiple spaces
            {"dense": "[0.1 0.2]"},  # space
            {"dense": "[0.1  0.2]"},  # multiple spaces
        ]

        tensors = (tensorizer.numberize(row) for row in rows)
        for tensor in tensors:
            self.assertEqual([0.1, 0.2], tensor)

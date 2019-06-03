#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest
from typing import List

import numpy as np
import torch
from pytext.data.sources.data_source import Gazetteer, SafeFileWrapper
from pytext.data.sources.tsv import SessionTSVDataSource, TSVDataSource
from pytext.data.tensorizers import (
    AnnotationNumberizer,
    ByteTensorizer,
    ByteTokenTensorizer,
    FloatListTensorizer,
    GazetteerTensorizer,
    LabelListTensorizer,
    LabelTensorizer,
    SeqTokenTensorizer,
    TokenTensorizer,
    initialize_tensorizers,
)
from pytext.utils.test import import_tests_module


tests_module = import_tests_module()


class ListTensorizersTest(unittest.TestCase):
    def setUp(self):
        self.data = SessionTSVDataSource(
            SafeFileWrapper(tests_module.test_file("seq_tagging_example.tsv")),
            field_names=["session_id", "intent", "goal", "label"],
            schema={"intent": List[str], "goal": List[str], "label": List[str]},
        )

    def test_initialize_list_tensorizers(self):
        tensorizers = {
            "intent": LabelListTensorizer(
                label_column="intent", pad_in_vocab=True, allow_unknown=True
            ),
            "goal": LabelListTensorizer(label_column="goal"),
        }
        initialize_tensorizers(tensorizers, self.data.train)
        self.assertEqual(9, len(tensorizers["intent"].vocab))
        self.assertEqual(7, len(tensorizers["goal"].vocab))

    def test_create_label_list_tensors(self):
        tensorizers = {
            "intent": LabelListTensorizer(
                label_column="intent", pad_in_vocab=True, allow_unknown=True
            )
        }
        initialize_tensorizers(tensorizers, self.data.train)
        tensors = [tensorizers["intent"].numberize(row) for row in self.data.train]
        # test label idx
        self.assertEqual([2, 3], tensors[0][0])
        self.assertEqual([4, 5], tensors[1][0])
        self.assertEqual([6, 7, 8], tensors[2][0])
        # test seq lens
        self.assertEqual(2, tensors[0][1])
        self.assertEqual(2, tensors[1][1])
        self.assertEqual(3, tensors[2][1])
        self.assertEqual(3, len(tensors))
        tensors, lens = tensorizers["intent"].tensorize(tensors)
        np.testing.assert_array_almost_equal(
            np.array([[2, 3, 1], [4, 5, 1], [6, 7, 8]]), tensors.detach().numpy()
        )
        np.testing.assert_array_almost_equal(np.array([2, 2, 3]), lens.detach().numpy())

    def test_label_list_tensors_no_pad_in_vocab(self):
        tensorizers = {
            "intent": LabelListTensorizer(
                label_column="intent", pad_in_vocab=False, allow_unknown=True
            )
        }
        initialize_tensorizers(tensorizers, self.data.train)
        self.assertEqual(8, len(tensorizers["intent"].vocab))
        tensors = []
        for row in self.data.train:
            row["intent"].append("unknown")
            tensors.append(tensorizers["intent"].numberize(row))
        tensors, lens = tensorizers["intent"].tensorize(tensors)
        np.testing.assert_array_almost_equal(
            np.array([[1, 2, 0, -1], [3, 4, 0, -1], [5, 6, 7, 0]]),
            tensors.detach().numpy(),
        )


# fmt: off
EXPECTED_ACTIONS = [
    [0, 1, 1, 2, 1, 3, 3],
    [4, 1, 5, 1, 1, 1, 3, 1, 6, 7, 8, 1, 3, 1, 3, 3, 1, 1, 9, 1, 1, 1, 1, 1, 3, 1, 3],
    [10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3],
    [0, 1, 1, 1, 1, 1, 3],
    [11, 1, 1, 1, 1, 1, 1, 1, 1, 1, 6, 7, 1, 3, 3, 1, 5, 12, 1, 3, 3, 1, 1, 1, 9, 1, 1, 1, 3, 1, 3],
    [4, 1, 1, 1, 1, 5, 1, 3, 1, 1, 13, 1, 1, 1, 3, 3],
    [4, 1, 1, 1, 1, 1, 5, 7, 1, 3, 3, 3],
    [14, 1, 1, 1, 6, 1, 3, 1, 5, 1, 3, 3],
    [0, 1, 1, 1, 1, 1, 2, 1, 3, 3],
    [10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3],
]
# fmt: on


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
        self.assertEqual(7, len(tensorizers["labels"].vocab))

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
        tokens, seq_len, token_ranges = next(tensors)
        self.assertEqual([24, 0, 0, 0], tokens)
        self.assertEqual(4, seq_len)
        self.assertEqual([(0, 1), (2, 6), (7, 11), (12, 18)], token_ranges)

        tokens, seq_len, token_ranges = next(tensors)
        self.assertEqual([13, 47, 9], tokens)
        self.assertEqual(3, seq_len)
        self.assertEqual([(0, 4), (5, 7), (8, 10)], token_ranges)

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

    def test_create_byte_token_tensors(self):
        tensorizer = ByteTokenTensorizer(
            text_column="text", max_seq_len=4, max_byte_len=5
        )
        # not initializing because initializing is a no-op for this tensorizer

        s1 = "I want some coffee today"
        s2 = "Turn it up"

        def ords(word, pad_to):
            return list(word.encode()) + [0] * (pad_to - len(word))

        batch = [{"text": s1}, {"text": s2}]
        # Note that the tokenizer lowercases here
        expected = [
            [ords("i", 5), ords("want", 5), ords("some", 5), ords("coffe", 5)],
            [ords("turn", 5), ords("it", 5), ords("up", 5), ords("", 5)],
        ]
        expected_token_lens = [4, 3]
        expected_byte_lens = [[1, 4, 4, 5], [4, 2, 2, 0]]

        bytes, token_lens, byte_lens = tensorizer.tensorize(
            [tensorizer.numberize(row) for row in batch]
        )
        self.assertIsInstance(bytes, torch.LongTensor)
        self.assertIsInstance(token_lens, torch.LongTensor)
        self.assertIsInstance(byte_lens, torch.LongTensor)
        self.assertEqual((2, 4, 5), bytes.size())
        self.assertEqual((2,), token_lens.size())
        self.assertEqual((2, 4), byte_lens.size())
        self.assertEqual(expected, bytes.tolist())
        self.assertEqual(expected_token_lens, token_lens.tolist())
        self.assertEqual(expected_byte_lens, byte_lens.tolist())

    def test_initialize_label_tensorizer(self):
        tensorizer = LabelTensorizer(label_column="label")
        init = tensorizer.initialize()
        init.send(None)  # kick
        for row in self.data.train:
            init.send(row)
        init.close()
        print(tensorizer.vocab._vocab)
        self.assertEqual(7, len(tensorizer.vocab))

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

    def test_gazetteer_tensor_bad_json(self):
        tensorizer = GazetteerTensorizer()

        data = TSVDataSource(
            train_file=SafeFileWrapper(
                tests_module.test_file("train_dict_features_bad_json.tsv")
            ),
            test_file=None,
            eval_file=None,
            field_names=["text", "dict"],
            schema={"text": str, "dict": Gazetteer},
        )

        init = tensorizer.initialize()
        init.send(None)  # kick
        with self.assertRaises(Exception):
            for row in data.train:
                init.send(row)
        init.close()

    def test_gazetteer_tensor(self):
        tensorizer = GazetteerTensorizer()

        data = TSVDataSource(
            train_file=SafeFileWrapper(
                tests_module.test_file("train_dict_features.tsv")
            ),
            test_file=None,
            eval_file=None,
            field_names=["text", "dict"],
            schema={"text": str, "dict": Gazetteer},
        )

        init = tensorizer.initialize()
        init.send(None)  # kick
        for row in data.train:
            init.send(row)
        init.close()
        # UNK + PAD + 5 labels
        self.assertEqual(7, len(tensorizer.vocab))

        # only two rows in test file:
        # "Order coffee from Starbucks please"
        # "Order some fries from McDonalds please"
        for i, row in enumerate(data.train):
            if i == 0:
                idx, weights, lens = tensorizer.numberize(row)
                self.assertEqual([1, 1, 2, 3, 1, 1, 4, 1, 1, 1], idx)
                self.assertEqual(
                    [0.0, 0.0, 0.8, 0.2, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], weights
                )
                self.assertEqual([1, 2, 1, 1, 1], lens)
            if i == 1:
                idx, weights, lens = tensorizer.numberize(row)
                self.assertEqual([1, 1, 5, 1, 6, 1], idx)
                self.assertEqual([0.0, 0.0, 1.0, 0.0, 1.0, 0.0], weights)
                self.assertEqual([1, 1, 1, 1, 1, 1], lens)

        feats, weights, lens = tensorizer.tensorize(
            tensorizer.numberize(row) for row in data.train
        )
        self.assertEqual(
            [
                [1, 1, 2, 3, 1, 1, 4, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 5, 1, 1, 1, 6, 1, 1, 1],
            ],
            feats.numpy().tolist(),
        )
        self.assertEqual(
            str(
                [
                    [0.0, 0.0, 0.8, 0.2, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                ]
            ),
            str(
                [[round(w, 2) for w in utt_weights] for utt_weights in weights.numpy()]
            ),
        )
        self.assertEqual(
            [[1, 2, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]], lens.numpy().tolist()
        )

    def test_seq_tensor(self):
        tensorizer = SeqTokenTensorizer()

        data = TSVDataSource(
            train_file=SafeFileWrapper(
                tests_module.test_file("train_seq_features.tsv")
            ),
            test_file=None,
            eval_file=None,
            field_names=["text_seq"],
            schema={"text_seq": List[str]},
        )

        init = tensorizer.initialize()
        init.send(None)  # kick
        for row in data.train:
            init.send(row)
        init.close()
        # UNK + PAD + 6 tokens
        self.assertEqual(8, len(tensorizer.vocab))

        # only one row in test file:
        # ["where do you wanna meet?", "MPK"]
        for row in data.train:
            idx, lens = tensorizer.numberize(row)
            self.assertEqual(2, lens)
            self.assertEqual([[2, 3, 4, 5, 6], [7, 1, 1, 1, 1]], idx)

    def test_seq_tensor_with_bos_eos_eol_bol(self):
        tensorizer = SeqTokenTensorizer(
            add_bos_token=True,
            add_eos_token=True,
            add_bol_token=True,
            add_eol_token=True,
        )

        data = TSVDataSource(
            train_file=SafeFileWrapper(
                tests_module.test_file("train_seq_features.tsv")
            ),
            test_file=None,
            eval_file=None,
            field_names=["text_seq"],
            schema={"text_seq": List[str]},
        )

        init = tensorizer.initialize()
        init.send(None)  # kick
        for row in data.train:
            init.send(row)
        init.close()
        # UNK + PAD + BOS + EOS + BOL + EOL + 6 tokens
        self.assertEqual(12, len(tensorizer.vocab))

        # only one row in test file:
        # ["where do you wanna meet?", "MPK"]
        for row in data.train:
            idx, lens = tensorizer.numberize(row)
            self.assertEqual(4, lens)
            self.assertEqual(
                [
                    [2, 4, 3, 1, 1, 1, 1],
                    [2, 6, 7, 8, 9, 10, 3],
                    [2, 11, 3, 1, 1, 1, 1],
                    [2, 5, 3, 1, 1, 1, 1],
                ],
                idx,
            )

    def test_create_float_list_tensor(self):
        tensorizer = FloatListTensorizer(column="dense", dim=2, error_check=True)
        rows = [
            {"dense": "[0.1,0.2]"},  # comma
            {"dense": "[0.1, 0.2]"},  # comma with single space
            {"dense": "[0.1,  0.2]"},  # comma with multiple spaces
            {"dense": "[0.1 0.2]"},  # space
            {"dense": "[0.1  0.2]"},  # multiple spaces
            {"dense": "[ 0.1  0.2]"},  # space after [
            {"dense": "[0.1  0.2 ]"},  # space before ]
        ]

        tensors = (tensorizer.numberize(row) for row in rows)
        for tensor in tensors:
            self.assertEqual([0.1, 0.2], tensor)

        # test that parsing 0. and 1. works
        a_row = {"dense": "[0.  1.]"}
        tensor = tensorizer.numberize(a_row)
        self.assertEqual([0.0, 1.0], tensor)

    def test_annotation_num(self):
        data = TSVDataSource(
            SafeFileWrapper(tests_module.test_file("compositional_seq2seq_unit.tsv")),
            test_file=None,
            eval_file=None,
            field_names=["text", "seqlogical"],
            schema={"text": str, "seqlogical": str},
        )
        nbrz = AnnotationNumberizer()
        init = nbrz.initialize()
        init.send(None)  # kick
        for row in data.train:
            init.send(row)
        init.close()

        # vocab = {'IN:GET_INFO_TRAFFIC': 0, 'SHIFT': 1, 'SL:LOCATION': 2,
        # 'REDUCE': 3, 'IN:GET_DIRECTIONS': 4, 'SL:DESTINATION': 5, 'SL:SOURCE': 6,
        # 'IN:GET_LOCATION_HOME': 7, 'SL:CONTACT': 8, 'SL:DATE_TIME_DEPARTURE': 9,
        # 'IN:UNSUPPORTED_NAVIGATION': 10, 'IN:GET_ESTIMATED_DURATION': 11,
        # 'IN:GET_LOCATION_WORK': 12, 'SL:PATH_AVOID': 13, 'IN:GET_DISTANCE': 14}

        self.assertEqual(15, len(nbrz.vocab))
        self.assertEqual(1, nbrz.shift_idx)
        self.assertEqual(3, nbrz.reduce_idx)
        self.assertEqual([10], nbrz.ignore_subNTs_roots)
        self.assertEqual(
            [0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], nbrz.valid_NT_idxs
        )
        self.assertEqual([0, 4, 7, 10, 11, 12, 14], nbrz.valid_IN_idxs)
        self.assertEqual([2, 5, 6, 8, 9, 13], nbrz.valid_SL_idxs)

        for row, expected in zip(data.train, EXPECTED_ACTIONS):
            actions = nbrz.numberize(row)
            self.assertEqual(expected, actions)

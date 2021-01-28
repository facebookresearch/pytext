#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

import torch
from pytext.torchscript.batchutils import (
    destructure_tensor_list,
    make_prediction_tokens,
    make_prediction_texts,
    make_prediction_texts_dense,
)


class BatchUtilsTest(unittest.TestCase):
    def test_make_prediction_tokens(self) -> None:
        # Negative Case: Empty request batch test in List[] format
        with self.assertRaises(RuntimeError):
            make_prediction_tokens([])

        # Negative Case: Empty request batch test in List[Tuple([List[]])] format
        with self.assertRaises(RuntimeError):
            make_prediction_tokens([([],)])

        # Negative Case: None as input
        with self.assertRaises(TypeError):
            make_prediction_tokens(None)

        # Negative Case: Bad batch token format with List[Tuple()]
        with self.assertRaises(IndexError):
            make_prediction_tokens([()])

        # Positve Case: Multiple tokens in one request batch
        multiple_tokens_request_batch = [
            (
                [
                    ["token_1_1"],
                    ["token_1_2_1", "token_1_2_2"],
                ],
            ),
            (
                [
                    ["token_2_1"],
                ],
            ),
        ]
        self.assertEqual(
            make_prediction_tokens(multiple_tokens_request_batch),
            [["token_1_1"], ["token_1_2_1", "token_1_2_2"], ["token_2_1"]],
        )

    def test_destructure_tensor_list_empty(self):
        tensor_list = [torch.tensor([0])]
        result_tensor_list = destructure_tensor_list([], tensor_list)
        self.assertEqual(len(result_tensor_list), 0)

    def test_destructure_tensor_list_short_complete(self):
        tensor_list = [torch.tensor([0]), torch.tensor([1, 2])]
        result_tensor_list = destructure_tensor_list([1, 1], tensor_list)
        self.assertEqual(len(result_tensor_list), 2)
        self.assertEqual(len(result_tensor_list[0]), 1)
        self.assertEqual(len(result_tensor_list[1]), 1)
        self.assertTrue(torch.equal(result_tensor_list[0][0], torch.tensor([0])))
        self.assertTrue(torch.equal(result_tensor_list[1][0], torch.tensor([1, 2])))

    def test_destructure_tensor_list_short_incomplete(self):
        tensor_list = [torch.tensor([0]), torch.tensor([1, 2, 3]), torch.tensor([4, 5])]
        result_tensor_list = destructure_tensor_list([1, 1, 0], tensor_list)
        self.assertEqual(len(result_tensor_list), 3)
        self.assertEqual(len(result_tensor_list[0]), 1)
        self.assertEqual(len(result_tensor_list[1]), 1)
        self.assertEqual(len(result_tensor_list[2]), 0)
        self.assertTrue(torch.equal(result_tensor_list[0][0], torch.tensor([0])))
        self.assertTrue(torch.equal(result_tensor_list[1][0], torch.tensor([1, 2, 3])))

    def test_destructure_tensor_list_long1(self):
        tensor_list = [torch.tensor([0]) for i in range(1000)]
        result_tensor_list = destructure_tensor_list([100, 200, 300, 400], tensor_list)
        self.assertEqual(len(result_tensor_list), 4)
        self.assertEqual(len(result_tensor_list[0]), 100)
        self.assertEqual(len(result_tensor_list[1]), 200)
        self.assertEqual(len(result_tensor_list[2]), 300)
        self.assertEqual(len(result_tensor_list[3]), 400)
        for sublist in result_tensor_list:
            for tens in sublist:
                self.assertTrue(torch.equal(tens, torch.tensor([0])))

    def test_destructure_tensor_list_long2(self):
        tensor_list = [torch.tensor([0]) for i in range(2000)]
        length_list = [1, 2, 3, 4] * 100
        result_tensor_list = destructure_tensor_list(length_list, tensor_list)
        self.assertEqual(len(result_tensor_list), len(length_list))
        for i in range(len(length_list)):
            self.assertEqual(len(result_tensor_list[i]), length_list[i])
        for sublist in result_tensor_list:
            for tens in sublist:
                self.assertTrue(torch.equal(tens, torch.tensor([0])))

    def test_destructure_tensor_list_long3(self):
        tensor_list = [torch.tensor([0] * i) for i in range(100)]
        length_list = [1, 2, 3, 4] * 10

        result_tensor_list = destructure_tensor_list(length_list, tensor_list)
        self.assertEqual(len(result_tensor_list), len(length_list))
        for i in range(len(length_list)):
            self.assertEqual(len(result_tensor_list[i]), length_list[i])
        i = 0
        for sublist in result_tensor_list:
            for tens in sublist:
                self.assertTrue(torch.equal(tens, torch.tensor([0] * i)))
                i += 1

    def test_make_prediction_texts(self) -> None:
        # Negative Case: Empty request batch test in List[] format
        with self.assertRaises(RuntimeError):
            make_prediction_texts([])

        # Negative Case: Empty request batch test in List[Tuple([List[str]])] format
        with self.assertRaises(RuntimeError):
            make_prediction_texts([([],)])

        # Negative Case: None as input
        with self.assertRaises(TypeError):
            make_prediction_texts(None)

        # Negative Case: Bad batch text format with List[Tuple()]
        with self.assertRaises(IndexError):
            make_prediction_texts([()])

        # Positve Case
        multiple_texts_request_batch = [
            (
                [
                    "First Line Text",
                    "Other Text",
                ],
            ),
            (
                [
                    "Second Line Text",
                ],
            ),
        ]
        self.assertEqual(
            make_prediction_texts(multiple_texts_request_batch),
            ["First Line Text", "Other Text", "Second Line Text"],
        )

    def test_make_prediction_texts_dense(self) -> None:
        # Negative Case: Empty request batch test in List[] format
        with self.assertRaises(RuntimeError):
            make_prediction_texts_dense([])

        # Negative Case: Empty request batch test in List[Tuple([List[], List[]])] format
        with self.assertRaises(RuntimeError):
            make_prediction_texts_dense(
                [
                    (
                        [],
                        [],
                    )
                ]
            )

        # Negative Case: None as input
        with self.assertRaises(TypeError):
            make_prediction_texts_dense(None)

        # Negative Case: Bad format with List[Tuple()]
        with self.assertRaises(IndexError):
            make_prediction_texts_dense([()])

        # Negative Case: texts/dense client batch length mismatch
        mismatch_request_batch = [
            (
                [
                    "First Line Text",
                    "Other Text",
                ],
                [
                    [1.0],
                ],
            ),
        ]
        with self.assertRaises(RuntimeError):
            make_prediction_texts_dense(mismatch_request_batch)

        # Positve Case
        multiple_text_dense_request_batch = [
            (
                [
                    "First Line Text",
                    "Other Text",
                ],
                [
                    [1.0],
                    [0.0],
                ],
            ),
            (
                [
                    "Second Line Text",
                ],
                [
                    [1.0],
                ],
            ),
        ]
        self.assertEqual(
            make_prediction_texts_dense(multiple_text_dense_request_batch),
            (
                ["First Line Text", "Other Text", "Second Line Text"],
                [[1.0], [0.0], [1.0]],
            ),
        )

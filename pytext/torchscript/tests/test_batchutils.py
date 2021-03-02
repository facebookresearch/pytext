#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

import torch
from pytext.torchscript.batchutils import (
    destructure_any_list,
    destructure_dict_list,
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

    def test_destructure_any_list_empty(self):
        # -1- two empty lists
        res_list = destructure_any_list([], [])
        self.assertEqual(len(res_list), 0)

        # -2- wrong types passed to the function
        client_batch = 0
        result_any_list = 0
        with self.assertRaises(TypeError):
            res_list = destructure_any_list(client_batch, result_any_list)

        # -3- 1st list is empty, 2nd list is not
        client_batch = []
        result_any_list = [1, 1, 1]
        res_list = destructure_any_list(client_batch, result_any_list)
        self.assertEqual(len(res_list), 0)

        # -4- 1st list is not empty, 2nd list is empty
        client_batch = [1, 1, 1]
        result_any_list = []
        print("***********case 4")
        res_list = destructure_any_list(client_batch, result_any_list)
        self.assertEqual(len(res_list), 3)

        # -5- both lists are not empty. 1st list contains range that exceeds the 2nd
        client_batch = [1, 3, 5]
        result_any_list = [0, 1, 2]
        print("***********case 5")
        res_list = destructure_any_list(client_batch, result_any_list)
        self.assertEqual(len(res_list), 3)
        self.assertEqual([0], res_list[0])
        self.assertEqual([1, 2], res_list[1])
        self.assertEqual([], res_list[2])

        # -6- both lists are not empty. 1st lists contains range does not exceed the 2nd.
        client_batch = [5, 3]
        result_any_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        res_list = destructure_any_list(client_batch, result_any_list)
        self.assertEqual(len(res_list), 2)
        self.assertEqual([0, 1, 2, 3, 4], res_list[0])
        self.assertEqual([5, 6, 7], res_list[1])

        # -7- client_batch ranges are 0 or negative. should raise runtime error
        client_batch = [4, 3, -1]
        result_any_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        with self.assertRaises(AssertionError):
            res_list = destructure_any_list(client_batch, result_any_list)

        client_batch = [3, 0, 5]
        with self.assertRaises(AssertionError):
            res_list = destructure_any_list(client_batch, result_any_list)

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

    def test_destructure_dict_list_empty(self):
        dict_list = [{"a": 0}]
        result_dict_list = destructure_dict_list([], dict_list)
        self.assertEqual(len(result_dict_list), 0)

    def test_destructure_dict_list_short_complete(self):
        dict_list = [{"a": 0}, {"b": 1, "c": 2}]
        result_dict_list = destructure_dict_list([1, 1], dict_list)
        self.assertEqual(len(result_dict_list), 2)
        self.assertEqual(len(result_dict_list[0]), 1)
        self.assertEqual(len(result_dict_list[1]), 1)
        self.assertEqual(result_dict_list[0][0], {"a": 0})
        self.assertEqual(result_dict_list[1][0], {"b": 1, "c": 2})

    def test_destructure_dict_list_short_incomplete(self):
        dict_list = [{"a": 0}, {"b": 1, "c": 2, "d": 3}, {"e": 4, "f": 5}]
        result_dict_list = destructure_dict_list([1, 1, 0], dict_list)
        self.assertEqual(len(result_dict_list), 3)
        self.assertEqual(len(result_dict_list[0]), 1)
        self.assertEqual(len(result_dict_list[1]), 1)
        self.assertEqual(len(result_dict_list[2]), 0)
        self.assertEqual(result_dict_list[0][0], {"a": 0})
        self.assertEqual(result_dict_list[1][0], {"b": 1, "c": 2, "d": 3})

    def test_destructure_dict_list_long1(self):
        dict_list = [{"a": 0} for i in range(1000)]
        result_dict_list = destructure_dict_list([100, 200, 300, 400], dict_list)
        self.assertEqual(len(result_dict_list), 4)
        self.assertEqual(len(result_dict_list[0]), 100)
        self.assertEqual(len(result_dict_list[1]), 200)
        self.assertEqual(len(result_dict_list[2]), 300)
        self.assertEqual(len(result_dict_list[3]), 400)
        for sublist in result_dict_list:
            for tens in sublist:
                self.assertEqual(tens, {"a": 0})

    def test_destructure_dict_list_long2(self):
        dict_list = [{"a": 0} for i in range(2000)]
        length_list = [1, 2, 3, 4] * 100
        result_dict_list = destructure_dict_list(length_list, dict_list)
        self.assertEqual(len(result_dict_list), len(length_list))
        for i in range(len(length_list)):
            self.assertEqual(len(result_dict_list[i]), length_list[i])
        for sublist in result_dict_list:
            for tens in sublist:
                self.assertEqual(tens, {"a": 0})

    def test_destructure_dict_list_long3(self):
        dict_list = [{"a": i} for i in range(100)]
        length_list = [1, 2, 3, 4] * 10

        result_dict_list = destructure_dict_list(length_list, dict_list)
        self.assertEqual(len(result_dict_list), len(length_list))
        for i in range(len(length_list)):
            self.assertEqual(len(result_dict_list[i]), length_list[i])
        i = 0
        for sublist in result_dict_list:
            for tens in sublist:
                self.assertEqual(tens, {"a": i})
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

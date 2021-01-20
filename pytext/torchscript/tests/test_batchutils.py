#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

import torch
from pytext.torchscript.batchutils import destructure_tensor_list


class BatchutilsTest(unittest.TestCase):
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

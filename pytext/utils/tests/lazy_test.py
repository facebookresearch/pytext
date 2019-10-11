#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

import torch
from pytext.utils import lazy
from torch import nn


class LazyTest(unittest.TestCase):
    def test_parameters_throws_exception_before_init(self):
        linear = lazy.Linear(4)
        with self.assertRaises(lazy.UninitializedLazyModuleError):
            list(linear.parameters())

        seq = nn.Sequential(linear)
        with self.assertRaises(lazy.UninitializedLazyModuleError):
            list(seq.parameters())

    def test_parameters_after_init(self):
        linear = lazy.Linear(4)
        linear = lazy.init_lazy_modules(linear, torch.rand(1, 2))
        self.assertEqual(2, len(list(linear.parameters())))

        seq = nn.Sequential(lazy.Linear(4))
        seq = lazy.init_lazy_modules(seq, torch.rand(1, 2))
        self.assertEqual(2, len(list(seq.parameters())))
        self.assertIsInstance(seq[0], nn.Linear)

    def test_lazy_linear(self):
        linear = lazy.Linear(4)
        input = torch.rand(1, 2)
        out = linear(input)
        self.assertEqual((1, 4), out.size())
        resolved = lazy.init_lazy_modules(linear, input)
        self.assertIsInstance(resolved, nn.Linear)
        self.assertEqual(2, resolved.in_features)
        self.assertEqual(4, resolved.out_features)
        self.assertIsNotNone(resolved.bias)
        self.assertTrue(torch.equal(out, resolved(input)))

    def test_lazy_linear_without_bais(self):
        linear = lazy.Linear(4, bias=False)
        input = torch.rand(1, 2)
        out = linear(input)
        self.assertEqual((1, 4), out.size())
        resolved = lazy.init_lazy_modules(linear, input)
        self.assertIsInstance(resolved, nn.Linear)
        self.assertEqual(2, resolved.in_features)
        self.assertEqual(4, resolved.out_features)
        self.assertFalse(resolved.bias)
        self.assertTrue(torch.equal(out, resolved(input)))

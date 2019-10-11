#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import random
import unittest

import torch
from pytext.models.representations.docnn import DocNNRepresentation as DocNN
from pytext.utils import lazy
from torch import nn


class DocNNTest(unittest.TestCase):
    def test_constructor_api(self):
        docnn = DocNN()
        self.assertIsInstance(docnn, DocNN)
        docnn_small_kernel = DocNN(kernel_num=10)
        self.assertIsInstance(docnn_small_kernel, DocNN)
        docnn_many_small_kernels = DocNN(kernel_num=5, kernel_sizes=list(range(1, 50)))
        self.assertIsInstance(docnn_many_small_kernels, DocNN)
        docnn_no_dropout = DocNN(dropout=0.0)
        self.assertIsInstance(docnn_no_dropout, DocNN)

    def test_from_config(self):
        docnn = DocNN.from_config(DocNN.Config(), embed_dim=None)
        self._test_docnn_component_with_variable_embedding(docnn)

        docnn_ignore_embed_dim = DocNN.from_config(DocNN.Config(), embed_dim=50)
        self._test_docnn_component_with_variable_embedding(docnn_ignore_embed_dim)

    def test_should_infer_embedding_dimension(self):
        docnn = DocNN()
        self._test_docnn_component_with_variable_embedding(docnn)

        docnn_small_kernel = DocNN(kernel_num=10)
        self._test_docnn_component_with_variable_embedding(docnn_small_kernel)

        docnn_many_small_kernels = DocNN(kernel_num=5, kernel_sizes=list(range(1, 50)))
        self._test_docnn_component_with_variable_embedding(docnn_many_small_kernels)

    def _test_docnn_component_with_variable_embedding(self, docnn):
        vocab_size = random.randint(50, 100)
        embedding_size = random.randint(15, 20)
        embedding = nn.Embedding(vocab_size, embedding_size)
        model = nn.Sequential(embedding, docnn)
        inputs_length_1 = random.randint(10, 25)
        inputs_1 = torch.LongTensor(
            [[random.randint(0, vocab_size - 1) for _ in range(inputs_length_1)]]
        )

        model = lazy.init_lazy_modules(model, inputs_1)

        results_1 = model(inputs_1)
        self.assertEqual([1, docnn.representation_dim], list(results_1.size()))

        # test that it's fine with different batch sizes
        inputs_length_2 = random.randint(10, 25)
        inputs_2 = torch.LongTensor(
            [
                [random.randint(0, vocab_size - 1) for _ in range(inputs_length_2)],
                [random.randint(0, vocab_size - 1) for _ in range(inputs_length_2)],
            ]
        )

        results_2 = model(inputs_2)
        self.assertEqual([2, docnn.representation_dim], list(results_2.size()))

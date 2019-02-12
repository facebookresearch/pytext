#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

import torch
from pytext.models.representations import bilstm
from torch import jit, nn


VOCAB_SIZE = 10
EMBEDDING_SIZE = 3


class BiLSTMTest(unittest.TestCase):
    def test_trace_bilstm_differ_batch_size(self):
        # BiLSTM torch tracing was using torch.new_zeros for default input hidden
        # states, which doesn't trace properly. torch.jit traces torch.new_zeros as
        # constant and therefore locks the traced model into a static batch size.
        # torch.LSTM now uses zeros, adding test case here to verify behavior.
        # see https://github.com/pytorch/pytorch/issues/16664

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_SIZE)
                self.bilstm = bilstm.BiLSTM(bilstm.BiLSTM.Config(), EMBEDDING_SIZE)

            def forward(self, tokens, seq_lengths):
                embeddings = self.embedding(tokens)
                return self.bilstm(embeddings, seq_lengths)

        model = Model()
        trace_inputs = (
            torch.LongTensor([[2, 3, 4], [2, 2, 1]]),
            torch.LongTensor([3, 2]),
        )

        trace = jit.trace(model, trace_inputs)

        test_inputs = (torch.LongTensor([[4, 5, 6]]), torch.LongTensor([3]))

        # we are just testing that this doesn't throw an exception
        trace(*test_inputs)

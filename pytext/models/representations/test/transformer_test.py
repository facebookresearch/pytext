#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import unittest

import torch
from pytext.models.representations.transformer import SentenceEncoder


class SentenceEncoderTest(unittest.TestCase):
    def test_extract_features(self):
        encoder = SentenceEncoder()
        tokens = torch.LongTensor([5, 10, 20])
        encoder.extract_features(tokens)

    def test_forward(self):
        encoder = SentenceEncoder()
        tokens = torch.LongTensor([5, 10, 20])
        encoder(tokens)

    def test_script(self):
        encoder = SentenceEncoder()
        scripted = torch.jit.script(encoder)
        tokens = torch.LongTensor([5, 10, 20])
        scripted.extract_features(tokens)
        encoder.eval()
        scripted.eval()
        self.assertTrue(torch.equal(encoder(tokens), scripted(tokens)))

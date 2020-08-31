#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import unittest

import torch
from pytext.models.representations.transformer import (
    SentenceEncoder,
    Transformer,
    TransformerLayer,
)


class SentenceEncoderTest(unittest.TestCase):
    def _small_encoder(self):
        layers = [TransformerLayer(embedding_dim=12) for _ in range(2)]
        transformer = Transformer(vocab_size=100, embedding_dim=12, layers=layers)
        return SentenceEncoder(transformer)

    def test_extract_features(self):
        encoder = self._small_encoder()
        tokens = torch.LongTensor([5, 10, 20])
        encoder.extract_features(tokens)

    def test_forward(self):
        encoder = self._small_encoder()
        tokens = torch.LongTensor([5, 10, 20])
        encoder(tokens)

    def test_script(self):
        encoder = self._small_encoder()
        scripted = torch.jit.script(encoder)
        tokens = torch.LongTensor([5, 10, 20])
        scripted.extract_features(tokens)
        encoder.eval()
        scripted.eval()

        encoder_out = encoder(tokens)
        scripted_out = scripted(tokens)
        assert len(encoder_out) == len(scripted_out)
        for encoder_tensor, scripted_tensor in zip(encoder_out, scripted_out):
            self.assertTrue(torch.equal(encoder_tensor, scripted_tensor))

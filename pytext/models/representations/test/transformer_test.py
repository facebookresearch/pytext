#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import unittest

import torch
from pytext.models.representations.transformer import (
    SentenceEncoder,
    Transformer,
    TransformerLayer,
)
from pytext.models.representations.transformer.representation import (
    TransformerRepresentation,
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


class SentenceRepresentationTest(unittest.TestCase):
    def test_representation(self):
        embedded_tokens = torch.rand(1, 3, 10)
        padding_mask = torch.BoolTensor([[False, False, True]])

        representation = TransformerRepresentation.from_config(
            TransformerRepresentation.Config(ffnn_embed_dim=10, num_attention_heads=2),
            embed_dim=10,
        )

        output = representation(embedded_tokens, padding_mask)

        self.assertEqual(output.shape, (1, 3, 10))

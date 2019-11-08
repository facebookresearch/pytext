#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import unittest

import torch
from pytext.models.representations.transformer_sentence_encoder import (
    TransformerSentenceEncoder,
)


class TransformerSentenceEncoderTest(unittest.TestCase):
    def setUp(self):
        self.batch_size = 10
        self.num_tokens = 20
        self.embedding_dim = 1024
        self.vocab_size = 1000
        self.padding_idx = 0
        self.num_encoder_layers = 6

        # Generate a tensor of token ids as input tokens
        self.tokens = (
            torch.randint(5, 1000, (self.batch_size, self.num_tokens))
        ).long()
        self.lengths = torch.tensor([self.num_tokens])
        self.pad_mask = (torch.ones(self.batch_size, self.num_tokens)).long()
        self.segment_labels = (torch.ones(self.batch_size, self.num_tokens)).long()
        self.positions = None

    def test_monolingual_transformer_sentence_encoder(self):

        input_tuple = (self.tokens, self.pad_mask, self.segment_labels, self.positions)

        sentence_encoder = TransformerSentenceEncoder.from_config(
            TransformerSentenceEncoder.Config(
                embedding_dim=self.embedding_dim,
                num_encoder_layers=self.num_encoder_layers,
                multilingual=False,
            ),
            output_encoded_layers=True,
            padding_idx=self.padding_idx,
            vocab_size=self.vocab_size,
        )

        encoded_layers, pooled_outputs = sentence_encoder(input_tuple)

        # Check sizes for pooled output
        self.assertEqual(pooled_outputs.size()[0], self.batch_size)
        self.assertEqual(pooled_outputs.size()[1], self.embedding_dim)

        # Check sizes for encoded_layers
        self.assertEqual(encoded_layers.__len__(), self.num_encoder_layers + 1)
        self.assertEqual(encoded_layers[-1].size()[0], self.batch_size)
        self.assertEqual(encoded_layers[-1].size()[1], self.num_tokens)
        self.assertEqual(encoded_layers[-1].size()[2], self.embedding_dim)

    def test_multilingual_transformer_sentence_encoder(self):

        input_tuple = (self.tokens, self.pad_mask, self.segment_labels, self.positions)

        sentence_encoder = TransformerSentenceEncoder.from_config(
            TransformerSentenceEncoder.Config(
                embedding_dim=self.embedding_dim,
                num_encoder_layers=self.num_encoder_layers,
                multilingual=True,
            ),
            output_encoded_layers=True,
            padding_idx=self.padding_idx,
            vocab_size=self.vocab_size,
        )

        encoded_layers, pooled_outputs = sentence_encoder(input_tuple)

        # Check sizes for pooled output
        self.assertEqual(pooled_outputs.size()[0], self.batch_size)
        self.assertEqual(pooled_outputs.size()[1], self.embedding_dim)

        # Check sizes for encoded_layers
        self.assertEqual(encoded_layers.__len__(), self.num_encoder_layers + 1)
        self.assertEqual(encoded_layers[-1].size()[0], self.batch_size)
        self.assertEqual(encoded_layers[-1].size()[1], self.num_tokens)
        self.assertEqual(encoded_layers[-1].size()[2], self.embedding_dim)

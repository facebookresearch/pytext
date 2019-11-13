#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

from pytext.data.tokenizers import GPT2BPETokenizer, SentencePieceTokenizer, Tokenizer
from pytext.data.tokenizers.tokenizer import Token


class TokenizeTest(unittest.TestCase):
    def test_tokenize(self):
        tokenizer = Tokenizer()
        sentence = "Order me a coffee"
        expected = ["order", "me", "a", "coffee"]
        tokens = tokenizer.tokenize(sentence)
        self.assertListEqual(expected, [t.value for t in tokens])

    def test_tokenize_dont_lowercase(self):
        tokenizer = Tokenizer(lowercase=False)
        sentence = "Order me a coffee"
        expected = ["Order", "me", "a", "coffee"]
        tokens = tokenizer.tokenize(sentence)
        self.assertListEqual(expected, [t.value for t in tokens])

    def test_split_with_regex(self):
        tokenizer = Tokenizer(split_regex=r"[\s,;!.?\"\(\)\-]+")
        sentence = """
            Your bones don't break, mine do. That's clear. Your cells react to
            bacteria and viruses differently than mine. You don't get sick,
            I do. That's also clear. But for some reason, you and I react the
            exact same way to water. We swallow it too fast, we choke. We get
            some in our lungs, we drown. However unreal it may seem, we are
            connected, you and I. We're on the same curve, just on opposite
            ends.
        """
        expected = """
            your bones don't break mine do that's clear your cells react to
            bacteria and viruses differently than mine you don't get sick
            i do that's also clear but for some reason you and i react the
            exact same way to water we swallow it too fast we choke we get
            some in our lungs we drown however unreal it may seem we are
            connected you and i we're on the same curve just on opposite ends
        """.split()
        tokens = tokenizer.tokenize(sentence)
        self.assertListEqual(expected, [t.value for t in tokens])

        sentence = '"Please, buy me a coffee?" He implored-in vain.'
        expected = "please buy me a coffee he implored in vain".split()
        tokens = tokenizer.tokenize(sentence)
        self.assertListEqual(expected, [t.value for t in tokens])


class GPT2BPETest(unittest.TestCase):
    def test_gpt2_bpe_tokenizer(self):
        text = "Prototype"
        expected = [Token("19703", 0, 4), Token("8690", 4, 9)]
        tokenizer = GPT2BPETokenizer.from_config(
            GPT2BPETokenizer.Config(
                bpe_vocab_path="pytext/data/test/data/gpt2_vocab.bpe",
                bpe_encoder_path="pytext/data/test/data/gpt2_encoder.json",
            )
        )
        tokens = tokenizer.tokenize(text)
        print(tokens)
        self.assertEqual(tokens, expected)


class SentencePieceTokenizerTest(unittest.TestCase):
    def test_tokenize(self):
        sentence = "Testing out sentencepiece"
        expected = [
            "▁T",
            "est",
            "ing",
            "▁out",
            "▁sen",
            "t",
            "ence",
            "p",
            "i",
            "e",
            "ce",
        ]
        sp_tokenizer = SentencePieceTokenizer.from_config(
            SentencePieceTokenizer.Config(
                sp_model_path="pytext/data/test/data/sentencepiece.model"
            )
        )
        tokens = sp_tokenizer.tokenize(sentence)
        tokens = [token.value for token in tokens]
        self.assertEqual(tokens, expected)

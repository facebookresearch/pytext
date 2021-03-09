#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

from pytext.data.tokenizers import (
    GPT2BPETokenizer,
    SentencePieceTokenizer,
    Tokenizer,
    WordPieceTokenizer,
)
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

    def test_tokenize_use_byte_offsets(self):
        tokenizer = Tokenizer(use_byte_offsets=True)
        sentence = "Ordér mê å ćoƒfee"
        expected = [
            Token("ordér", 0, 6),
            Token("mê", 7, 10),
            Token("å", 11, 13),
            Token("ćoƒfee", 14, 22),
        ]
        tokens = tokenizer.tokenize(sentence)
        self.assertListEqual(expected, tokens)

    def test_tokenize_no_byte_offsets(self):
        tokenizer = Tokenizer()
        sentence = "Ordér mê å ćoƒfee"
        expected = [
            Token("ordér", 0, 5),
            Token("mê", 6, 8),
            Token("å", 9, 10),
            Token("ćoƒfee", 11, 17),
        ]
        tokens = tokenizer.tokenize(sentence)
        self.assertListEqual(expected, tokens)

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


class WordpieceTokenizerTest(unittest.TestCase):
    def test_wordpiece_tokenizer(self):
        text = "Marcó Lopᚠz"
        expected = [
            Token("m", 0, 1),
            Token("##ar", 1, 3),
            Token("##c", 3, 4),
            Token(value="##o", start=4, end=5),
            Token(value="[UNK]", start=6, end=11),
        ]
        tokenizer = WordPieceTokenizer.from_config(
            WordPieceTokenizer.Config(
                wordpiece_vocab_path="pytext/data/test/data/wordpiece_1k.txt"
            )
        )
        tokens = tokenizer.tokenize(text)
        print(tokens)
        self.assertEqual(tokens, expected)


class GPT2BPETest(unittest.TestCase):
    def test_gpt2_bpe_tokenizer(self):
        tokenizer = GPT2BPETokenizer.from_config(
            GPT2BPETokenizer.Config(
                bpe_vocab_path="pytext/data/test/data/gpt2_vocab.bpe",
                bpe_encoder_path="pytext/data/test/data/gpt2_encoder.json",
            )
        )
        text_list = ["Prototype", " Prototype"]
        expected_list = [
            [Token("19703", 0, 4), Token("8690", 4, 9)],
            [Token("220", 0, 0), Token("19703", 1, 5), Token("8690", 5, 10)],
        ]

        for (text, expected) in zip(text_list, expected_list):
            tokens = tokenizer.tokenize(text)
            self.assertEqual(tokens, expected)


class SentencePieceTokenizerTest(unittest.TestCase):
    def test_tokenize(self):
        sentence = "Testing out sentencepiece"
        expected = [
            Token(value="▁T", start=0, end=1),
            Token(value="est", start=1, end=4),
            Token(value="ing", start=4, end=7),
            Token(value="▁out", start=8, end=11),
            Token(value="▁sen", start=12, end=15),
            Token(value="t", start=15, end=16),
            Token(value="ence", start=16, end=20),
            Token(value="p", start=20, end=21),
            Token(value="i", start=21, end=22),
            Token(value="e", start=22, end=23),
            Token(value="ce", start=23, end=25),
        ]
        sp_tokenizer = SentencePieceTokenizer.from_config(
            SentencePieceTokenizer.Config(
                sp_model_path="pytext/data/test/data/sentencepiece.model"
            )
        )
        tokens = sp_tokenizer.tokenize(sentence)
        self.assertEqual(tokens, expected)

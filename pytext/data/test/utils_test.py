#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

from pytext.data import utils


class TargetTest(unittest.TestCase):
    def test_align_target_label(self):
        target = [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
        label_list = ["l1", "l2", "l3"]
        batch_label_list = [["l3", "l2", "l1"], ["l1", "l3", "l2"]]
        align_target = utils.align_target_label(target, label_list, batch_label_list)
        self.assertListEqual(align_target, [[0.3, 0.2, 0.1], [0.1, 0.3, 0.2]])


class PaddingTest(unittest.TestCase):
    def testPadding(self):
        self.assertEqual(
            [[1, 2, 3], [1, 0, 0]], utils.pad([[1, 2, 3], [1]], pad_token=0)
        )
        self.assertEqual(
            [[[1], [2], [3]], [[1], [0], [0]]],
            utils.pad([[[1], [2], [3]], [[1]]], pad_token=0),
        )
        self.assertEqual(
            [[1, 2, 3, 4, 5, 6, 7], [9, 9, 9, 9, 9, 9, 9]],
            utils.pad([[1, 2, 3, 4, 5, 6, 7], []], pad_token=9),
        )

    def testPaddingProvideShape(self):
        self.assertEqual(
            [[0, 0, 0], [0, 0, 0]], utils.pad([], pad_token=0, pad_shape=(2, 3))
        )
        self.assertEqual(
            [[1, 2, 3], [1, 0, 0]],
            utils.pad([[1, 2, 3], [1]], pad_token=0, pad_shape=(2, 3)),
        )
        self.assertEqual([], utils.pad([], pad_token=0, pad_shape=()))


class TokenizeTest(unittest.TestCase):
    def test_tokenize(self):
        tokenizer = utils.Tokenizer()
        sentence = "Order me a coffee"
        expected = ["order", "me", "a", "coffee"]
        tokens = tokenizer.tokenize(sentence)
        self.assertListEqual(expected, [t.value for t in tokens])

    def test_tokenize_dont_lowercase(self):
        tokenizer = utils.Tokenizer(lowercase=False)
        sentence = "Order me a coffee"
        expected = ["Order", "me", "a", "coffee"]
        tokens = tokenizer.tokenize(sentence)
        self.assertListEqual(expected, [t.value for t in tokens])

    def test_split_with_regex(self):
        tokenizer = utils.Tokenizer(split_regex=r"[\s,;!.?\"\(\)\-]+")
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


class VocabularyTest(unittest.TestCase):
    def testBuildVocabulary(self):
        tokens = """
            your bones don't break mine do that's clear your cells react to
            bacteria and viruses differently than mine you don't get sick
            i do that's also clear but for some reason you and i react the
            exact same way to water we swallow it too fast we choke we get
            some in our lungs we drown however unreal it may seem we are
            connected you and i we're on the same curve just on opposite ends
        """.split()
        builder = utils.VocabBuilder()
        builder.add_all(tokens)
        vocab = builder.make_vocab()
        self.assertEqual(54, len(vocab))

        indices = vocab.lookup_all(["can i get a coffee".split()])
        self.assertEqual([[0, 21, 19, 0, 0]], indices)

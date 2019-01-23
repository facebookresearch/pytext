#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import unittest

from pytext.config.field_config import FeatureConfig
from pytext.data.featurizer import InputRecord, SimpleFeaturizer


class SimpleFeaturizerTest(unittest.TestCase):
    def setUp(self):
        self.sentence = "Order me a coffee"

    def test_tokenize(self):
        featurizer = SimpleFeaturizer.from_config(
            SimpleFeaturizer.Config(), FeatureConfig()
        )
        features = featurizer.featurize(InputRecord(raw_text=self.sentence))
        expected_tokens = ["order", "me", "a", "coffee"]
        expected_chars = [list(tok) for tok in expected_tokens]
        self.assertListEqual(features.tokens, expected_tokens)
        self.assertListEqual(features.characters, expected_chars)

    def test_tokenize_dont_lowercase(self):
        featurizer = SimpleFeaturizer.from_config(
            SimpleFeaturizer.Config(lowercase_tokens=False), FeatureConfig()
        )
        features = featurizer.featurize(InputRecord(raw_text=self.sentence))
        expected_tokens = ["Order", "me", "a", "coffee"]
        expected_chars = [list(tok) for tok in expected_tokens]
        self.assertListEqual(features.tokens, expected_tokens)
        self.assertListEqual(features.characters, expected_chars)

    def test_convert_to_bytes(self):
        featurizer = SimpleFeaturizer.from_config(
            SimpleFeaturizer.Config(convert_to_bytes=True, lowercase_tokens=False),
            FeatureConfig(),
        )
        features = featurizer.featurize(InputRecord(raw_text=self.sentence))
        expected_tokens = list("Order me a coffee")
        expected_chars = [list(char) for char in expected_tokens]
        self.assertListEqual(features.tokens, expected_tokens)
        self.assertListEqual(features.characters, expected_chars)

    def test_tokenize_add_sentence_markers(self):
        featurizer = SimpleFeaturizer.from_config(
            SimpleFeaturizer.Config(sentence_markers=("<s>", "</s>")), FeatureConfig()
        )
        tokens = featurizer.featurize(InputRecord(raw_text=self.sentence)).tokens
        self.assertListEqual(tokens, ["<s>", "order", "me", "a", "coffee", "</s>"])

    def test_split_with_regex(self):
        featurizer = SimpleFeaturizer.from_config(
            SimpleFeaturizer.Config(split_regex=r"[\s,;!.?\"\(\)\-]+"), FeatureConfig()
        )
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
        tokens = featurizer.featurize(InputRecord(raw_text=sentence)).tokens
        self.assertListEqual(expected, tokens)

        sentence = '"Please, buy me a coffee?" He implored-in vain.'
        expected = "please buy me a coffee he implored in vain".split()
        tokens = featurizer.featurize(InputRecord(raw_text=sentence)).tokens
        self.assertListEqual(expected, tokens)

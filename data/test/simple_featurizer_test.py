#!/usr/bin/env python3
import unittest

from pytext.config.field_config import FeatureConfig
from pytext.data.featurizer import InputRecord, SimpleFeaturizer


class SimpleFeaturizerTest(unittest.TestCase):
    def setUp(self):
        self.featurizer = SimpleFeaturizer.from_config(
            SimpleFeaturizer.Config(), FeatureConfig()
        )
        self.sentence = "Order me a coffee"

    def test_tokenize(self):
        # Lower case
        tokens = self.featurizer.featurize(InputRecord(raw_text=self.sentence)).tokens
        self.assertListEqual(tokens, ["order", "me", "a", "coffee"])

        # Don't lower cause
        self.featurizer.lowercase_tokens = False
        tokens = self.featurizer.featurize(InputRecord(raw_text=self.sentence)).tokens
        self.assertListEqual(tokens, ["Order", "me", "a", "coffee"])

        # Add sentence markers
        self.featurizer.sentence_markers = ("<s>", "</s>")
        tokens = self.featurizer.featurize(InputRecord(raw_text=self.sentence)).tokens
        self.assertListEqual(tokens, ["<s>", "Order", "me", "a", "coffee", "</s>"])

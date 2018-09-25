#!/usr/bin/env python3

import unittest

from facebook.assistant.lib.featurization_lib import DEFAULT_LOCALE
from pytext.data.featurizer import InputRecord
from pytext.data.shared_featurizer import SharedFeaturizer


TEST_UTTERANCE = "The quick brown fox jumped over the lazy dog"
INIT_TOKEN = "<s>"
EOS_TOKEN = "</s>"


class SharedFeaturizerTest(unittest.TestCase):
    def test_tokenize_raw_text(self):
        """Just test that API calling works without breaking.
           Actual funcitonal testing is covered in Featurizer's test."""
        featurizer = SharedFeaturizer(lowercase_tokens=False)
        tokenized_text = featurizer.tokenize(InputRecord(TEST_UTTERANCE))
        expected_tokens = TEST_UTTERANCE.split()
        self.assertEqual(len(tokenized_text.tokens), len(expected_tokens))
        for i in range(len(expected_tokens)):
            self.assertEqual(tokenized_text.tokens[i], expected_tokens[i])

    def test_featurize_no_sentence_markers(self):
        featurizer = SharedFeaturizer()
        model_feats = featurizer.featurize(InputRecord(TEST_UTTERANCE))
        expected_tokens = TEST_UTTERANCE.split()
        self.assertEqual(len(model_feats.tokens), len(expected_tokens))
        for i in range(len(expected_tokens)):
            self.assertEqual(model_feats.tokens[i], expected_tokens[i].lower())

    def test_featurize_sentence_markers(self):
        featurizer = SharedFeaturizer(
            sentence_markers_dict={DEFAULT_LOCALE: (INIT_TOKEN, EOS_TOKEN)}
        )
        model_feats = featurizer.featurize(InputRecord(TEST_UTTERANCE))
        expected_tokens = TEST_UTTERANCE.split()
        self.assertEqual(len(model_feats.tokens), len(expected_tokens) + 2)
        self.assertEqual(model_feats.tokens[0], INIT_TOKEN)
        self.assertEqual(model_feats.tokens[-1], EOS_TOKEN)
        for i in range(len(expected_tokens)):
            self.assertEqual(model_feats.tokens[i + 1], expected_tokens[i].lower())

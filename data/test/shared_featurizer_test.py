#!/usr/bin/env python3

import unittest

from facebook.assistant.lib.featurization_lib import DEFAULT_LOCALE
from pytext.data.shared_featurizer import SharedFeaturizer


TEST_UTTERANCE = "the quick brown fox jumped over the lazy dog"
INIT_TOKEN = "<s>"
EOS_TOKEN = "</s>"


class SharedFeaturizerTest(unittest.TestCase):
    def test_featurize_no_sentence_markers(self):
        featurizer = SharedFeaturizer()
        model_feats = featurizer.featurize(TEST_UTTERANCE, "")
        tokens = TEST_UTTERANCE.split()
        self.assertEqual(len(model_feats.tokens), len(tokens))
        for i in range(len(tokens)):
            self.assertEqual(model_feats.tokens[i], tokens[i])

    def test_featurize_sentence_markers(self):
        featurizer = SharedFeaturizer(
            sentence_markers_dict={DEFAULT_LOCALE: (INIT_TOKEN, EOS_TOKEN)}
        )
        model_feats = featurizer.featurize(TEST_UTTERANCE, "")
        tokens = TEST_UTTERANCE.split()
        self.assertEqual(len(model_feats.tokens), len(tokens) + 2)
        self.assertEqual(model_feats.tokens[0], INIT_TOKEN)
        self.assertEqual(model_feats.tokens[-1], EOS_TOKEN)
        for i in range(len(tokens)):
            self.assertEqual(model_feats.tokens[i + 1], tokens[i])

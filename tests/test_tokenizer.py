#!/usr/bin/env python3

from libfb.py import testutil
from pytext.shared_tokenizer import SharedTokenizer

tokenizer = SharedTokenizer()
sentences = [
    ("how are you", [("how", (0, 3)), ("are", (4, 7)), ("you", (8, 11))]),
    ("i'm trying", [("i'm", (0, 3)), ("trying", (4, 10))]),
]


class TestTokenizer(testutil.BaseFacebookTestCase):
    def test_tokenize(self):
        for (sentence, oracle_toks_ranges) in sentences:
            tokens = tokenizer.tokenize(sentence)
            oracle_toks = [x[0] for x in oracle_toks_ranges]
            self.assertEqual(tokens, oracle_toks)

    def test_tokenizeranges(self):
        for (sentence, oracle_toks_ranges) in sentences:
            tokens_ranges = tokenizer.tokenize_with_ranges(sentence)
            self.assertEqual(tokens_ranges, oracle_toks_ranges)

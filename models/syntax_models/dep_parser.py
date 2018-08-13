#!/usr/bin/env python3

import torch.nn as nn
from pytext.shared_tokenizer import SharedTokenizer
from pytext.tools.spacy_nlp import SpacyNLP


class DepParser(nn.Module):
    def __init__(self, config):
        super().__init__()
        tokenizer = SharedTokenizer()
        self.spacy_nlp = SpacyNLP(tokenizer, 1)

    def forward(self, texts):
        return self.spacy_nlp.annotate(texts)

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from .babi import BABI20
from .imdb import IMDB
from .language_modeling import (
    LanguageModelingDataset,
    WikiText2,
    WikiText103,
    PennTreebank,
)  # NOQA
from .nli import SNLI, MultiNLI, XNLI
from .sequence_tagging import SequenceTaggingDataset, UDPOS, CoNLL2000Chunking  # NOQA
from .sst import SST
from .text_classification import (
    TextClassificationDataset,
    AG_NEWS,
    SogouNews,
    DBpedia,
    YelpReviewPolarity,
    YelpReviewFull,
    YahooAnswers,
    AmazonReviewPolarity,
    AmazonReviewFull,
)
from .translation import TranslationDataset, Multi30k, IWSLT, WMT14  # NOQA
from .trec import TREC
from .unsupervised_learning import EnWik9

__all__ = [
    "LanguageModelingDataset",
    "SNLI",
    "MultiNLI",
    "XNLI",
    "SST",
    "TranslationDataset",
    "Multi30k",
    "IWSLT",
    "WMT14",
    "WikiText2",
    "WikiText103",
    "PennTreebank",
    "TREC",
    "IMDB",
    "SequenceTaggingDataset",
    "UDPOS",
    "CoNLL2000Chunking",
    "BABI20",
    "TextClassificationDataset",
    "AG_NEWS",
    "SogouNews",
    "DBpedia",
    "YelpReviewPolarity",
    "YelpReviewFull",
    "YahooAnswers",
    "AmazonReviewPolarity",
    "AmazonReviewFull",
    "EnWik9",
]

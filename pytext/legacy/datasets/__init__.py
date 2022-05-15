#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from .babi import BABI20
from .imdb import IMDB
from .language_modeling import (
    LanguageModelingDataset,
    PennTreebank,
    WikiText103,
    WikiText2,
)  # NOQA
from .nli import MultiNLI, SNLI, XNLI
from .sequence_tagging import CoNLL2000Chunking, SequenceTaggingDataset, UDPOS  # NOQA
from .sst import SST
from .text_classification import (
    AG_NEWS,
    AmazonReviewFull,
    AmazonReviewPolarity,
    DBpedia,
    SogouNews,
    TextClassificationDataset,
    YahooAnswers,
    YelpReviewFull,
    YelpReviewPolarity,
)
from .translation import IWSLT, Multi30k, TranslationDataset, WMT14  # NOQA
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

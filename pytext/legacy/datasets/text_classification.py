#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import csv
import io
import logging

import torch
from torchtext.data.utils import get_tokenizer, ngrams_iterator
from torchtext.utils import download_from_url, extract_archive
from tqdm import tqdm

from ..vocab import build_vocab_from_iterator, Vocab

URLS = {
    "AG_NEWS": "https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbUDNpeUdjb0wxRms",
    "SogouNews": "https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbUkVqNEszd0pHaFE",
    "DBpedia": "https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbQ2Vic1kxMmZZQ1k",
    "YelpReviewPolarity": "https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbNUpYQ2N3SGlFaDg",
    "YelpReviewFull": "https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbZlU4dXhHTFhZQU0",
    "YahooAnswers": "https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9Qhbd2JNdDBsQUdocVU",
    "AmazonReviewPolarity": "https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbaW12WVVZS2drcnM",
    "AmazonReviewFull": "https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbZVhsUnRWRDhETzA",
}


def _csv_iterator(data_path, ngrams, yield_cls=False):
    tokenizer = get_tokenizer("basic_english")
    with io.open(data_path, encoding="utf8") as f:
        reader = csv.reader(f)
        for row in reader:
            tokens = " ".join(row[1:])
            tokens = tokenizer(tokens)
            if yield_cls:
                yield int(row[0]) - 1, ngrams_iterator(tokens, ngrams)
            else:
                yield ngrams_iterator(tokens, ngrams)


def _create_data_from_iterator(vocab, iterator, include_unk):
    data = []
    labels = []
    with tqdm(unit_scale=0, unit="lines") as t:
        for cls, tokens in iterator:
            if include_unk:
                tokens = torch.tensor([vocab[token] for token in tokens])
            else:
                token_ids = list(
                    filter(
                        lambda x: x is not Vocab.UNK, [vocab[token] for token in tokens]
                    )
                )
                tokens = torch.tensor(token_ids)
            if len(tokens) == 0:
                logging.info("Row contains no tokens.")
            data.append((cls, tokens))
            labels.append(cls)
            t.update(1)
    return data, set(labels)


class TextClassificationDataset(torch.utils.data.Dataset):
    """Defines an abstract text classification datasets.
    Currently, we only support the following datasets:

          - AG_NEWS
          - SogouNews
          - DBpedia
          - YelpReviewPolarity
          - YelpReviewFull
          - YahooAnswers
          - AmazonReviewPolarity
          - AmazonReviewFull

    """

    def __init__(self, vocab, data, labels):
        """Initiate text-classification dataset.

        Args:
            vocab: Vocabulary object used for dataset.
            data: a list of label/tokens tuple. tokens are a tensor after
                numericalizing the string tokens. label is an integer.
                [(label1, tokens1), (label2, tokens2), (label2, tokens3)]
            label: a set of the labels.
                {label1, label2}

        Examples:
            See the examples in examples/text_classification/

        """

        super(TextClassificationDataset, self).__init__()
        self._data = data
        self._labels = labels
        self._vocab = vocab

    def __getitem__(self, i):
        return self._data[i]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        for x in self._data:
            yield x

    def get_labels(self):
        return self._labels

    def get_vocab(self):
        return self._vocab


def _setup_datasets(
    dataset_name, root=".data", ngrams=1, vocab=None, include_unk=False
):
    dataset_tar = download_from_url(URLS[dataset_name], root=root)
    extracted_files = extract_archive(dataset_tar)

    for fname in extracted_files:
        if fname.endswith("train.csv"):
            train_csv_path = fname
        if fname.endswith("test.csv"):
            test_csv_path = fname

    if vocab is None:
        logging.info("Building Vocab based on {}".format(train_csv_path))
        vocab = build_vocab_from_iterator(_csv_iterator(train_csv_path, ngrams))
    else:
        if not isinstance(vocab, Vocab):
            raise TypeError("Passed vocabulary is not of type Vocab")
    logging.info("Vocab has {} entries".format(len(vocab)))
    logging.info("Creating training data")
    train_data, train_labels = _create_data_from_iterator(
        vocab, _csv_iterator(train_csv_path, ngrams, yield_cls=True), include_unk
    )
    logging.info("Creating testing data")
    test_data, test_labels = _create_data_from_iterator(
        vocab, _csv_iterator(test_csv_path, ngrams, yield_cls=True), include_unk
    )
    if len(train_labels ^ test_labels) > 0:
        raise ValueError("Training and test labels don't match")
    return (
        TextClassificationDataset(vocab, train_data, train_labels),
        TextClassificationDataset(vocab, test_data, test_labels),
    )


def AG_NEWS(*args, **kwargs):
    """Defines AG_NEWS datasets.

    The labels include:

        - 0 : World
        - 1 : Sports
        - 2 : Business
        - 3 : Sci/Tech

    Create supervised learning dataset: AG_NEWS

    Separately returns the training and test dataset

    Args:
        root: Directory where the datasets are saved. Default: ".data"
        ngrams: a contiguous sequence of n items from s string text.
            Default: 1
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        include_unk: include unknown token in the data (Default: False)

    Examples:
        >>> train_dataset, test_dataset = torchtext.datasets.AG_NEWS(ngrams=3)

    """

    return _setup_datasets(*(("AG_NEWS",) + args), **kwargs)


def SogouNews(*args, **kwargs):
    """Defines SogouNews datasets.

    The labels include:

        - 0 : Sports
        - 1 : Finance
        - 2 : Entertainment
        - 3 : Automobile
        - 4 : Technology

    Create supervised learning dataset: SogouNews

    Separately returns the training and test dataset

    Args:
        root: Directory where the datasets are saved. Default: ".data"
        ngrams: a contiguous sequence of n items from s string text.
            Default: 1
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        include_unk: include unknown token in the data (Default: False)

    Examples:
        >>> train_dataset, test_dataset = torchtext.datasets.SogouNews(ngrams=3)

    """

    return _setup_datasets(*(("SogouNews",) + args), **kwargs)


def DBpedia(*args, **kwargs):
    """Defines DBpedia datasets.

    The labels include:

        - 0 : Company
        - 1 : EducationalInstitution
        - 2 : Artist
        - 3 : Athlete
        - 4 : OfficeHolder
        - 5 : MeanOfTransportation
        - 6 : Building
        - 7 : NaturalPlace
        - 8 : Village
        - 9 : Animal
        - 10 : Plant
        - 11 : Album
        - 12 : Film
        - 13 : WrittenWork

    Create supervised learning dataset: DBpedia

    Separately returns the training and test dataset

    Args:
        root: Directory where the datasets are saved. Default: ".data"
        ngrams: a contiguous sequence of n items from s string text.
            Default: 1
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        include_unk: include unknown token in the data (Default: False)

    Examples:
        >>> train_dataset, test_dataset = torchtext.datasets.DBpedia(ngrams=3)

    """

    return _setup_datasets(*(("DBpedia",) + args), **kwargs)


def YelpReviewPolarity(*args, **kwargs):
    """Defines YelpReviewPolarity datasets.

    The labels include:

        - 0 : Negative polarity.
        - 1 : Positive polarity.

    Create supervised learning dataset: YelpReviewPolarity

    Separately returns the training and test dataset

    Args:
        root: Directory where the datasets are saved. Default: ".data"
        ngrams: a contiguous sequence of n items from s string text.
            Default: 1
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        include_unk: include unknown token in the data (Default: False)

    Examples:
        >>> train_dataset, test_dataset = torchtext.datasets.YelpReviewPolarity(ngrams=3)

    """

    return _setup_datasets(*(("YelpReviewPolarity",) + args), **kwargs)


def YelpReviewFull(*args, **kwargs):
    """Defines YelpReviewFull datasets.

    The labels include:

        0 - 4 : rating classes (4 is highly recommended).

    Create supervised learning dataset: YelpReviewFull

    Separately returns the training and test dataset

    Args:
        root: Directory where the datasets are saved. Default: ".data"
        ngrams: a contiguous sequence of n items from s string text.
            Default: 1
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        include_unk: include unknown token in the data (Default: False)

    Examples:
        >>> train_dataset, test_dataset = torchtext.datasets.YelpReviewFull(ngrams=3)

    """

    return _setup_datasets(*(("YelpReviewFull",) + args), **kwargs)


def YahooAnswers(*args, **kwargs):
    """Defines YahooAnswers datasets.

    The labels include:

        - 0 : Society & Culture
        - 1 : Science & Mathematics
        - 2 : Health
        - 3 : Education & Reference
        - 4 : Computers & Internet
        - 5 : Sports
        - 6 : Business & Finance
        - 7 : Entertainment & Music
        - 8 : Family & Relationships
        - 9 : Politics & Government

    Create supervised learning dataset: YahooAnswers

    Separately returns the training and test dataset

    Args:
        root: Directory where the datasets are saved. Default: ".data"
        ngrams: a contiguous sequence of n items from s string text.
            Default: 1
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        include_unk: include unknown token in the data (Default: False)

    Examples:
        >>> train_dataset, test_dataset = torchtext.datasets.YahooAnswers(ngrams=3)

    """

    return _setup_datasets(*(("YahooAnswers",) + args), **kwargs)


def AmazonReviewPolarity(*args, **kwargs):
    """Defines AmazonReviewPolarity datasets.

    The labels include:

        - 0 : Negative polarity
        - 1 : Positive polarity

    Create supervised learning dataset: AmazonReviewPolarity

    Separately returns the training and test dataset

    Args:
        root: Directory where the datasets are saved. Default: ".data"
        ngrams: a contiguous sequence of n items from s string text.
            Default: 1
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        include_unk: include unknown token in the data (Default: False)

    Examples:
       >>> train_dataset, test_dataset = torchtext.datasets.AmazonReviewPolarity(ngrams=3)

    """

    return _setup_datasets(*(("AmazonReviewPolarity",) + args), **kwargs)


def AmazonReviewFull(*args, **kwargs):
    """Defines AmazonReviewFull datasets.

    The labels include:

        0 - 4 : rating classes (4 is highly recommended)

    Create supervised learning dataset: AmazonReviewFull

    Separately returns the training and test dataset

    Args:
        root: Directory where the dataset are saved. Default: ".data"
        ngrams: a contiguous sequence of n items from s string text.
            Default: 1
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        include_unk: include unknown token in the data (Default: False)

    Examples:
        >>> train_dataset, test_dataset = torchtext.datasets.AmazonReviewFull(ngrams=3)

    """

    return _setup_datasets(*(("AmazonReviewFull",) + args), **kwargs)


DATASETS = {
    "AG_NEWS": AG_NEWS,
    "SogouNews": SogouNews,
    "DBpedia": DBpedia,
    "YelpReviewPolarity": YelpReviewPolarity,
    "YelpReviewFull": YelpReviewFull,
    "YahooAnswers": YahooAnswers,
    "AmazonReviewPolarity": AmazonReviewPolarity,
    "AmazonReviewFull": AmazonReviewFull,
}


LABELS = {
    "AG_NEWS": {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"},
    "SogouNews": {
        0: "Sports",
        1: "Finance",
        2: "Entertainment",
        3: "Automobile",
        4: "Technology",
    },
    "DBpedia": {
        0: "Company",
        1: "EducationalInstitution",
        2: "Artist",
        3: "Athlete",
        4: "OfficeHolder",
        5: "MeanOfTransportation",
        6: "Building",
        7: "NaturalPlace",
        8: "Village",
        9: "Animal",
        10: "Plant",
        11: "Album",
        12: "Film",
        13: "WrittenWork",
    },
    "YelpReviewPolarity": {0: "Negative polarity", 1: "Positive polarity"},
    "YelpReviewFull": {
        0: "score 1",
        1: "score 2",
        2: "score 3",
        3: "score 4",
        4: "score 5",
    },
    "YahooAnswers": {
        0: "Society & Culture",
        1: "Science & Mathematics",
        2: "Health",
        3: "Education & Reference",
        4: "Computers & Internet",
        5: "Sports",
        6: "Business & Finance",
        7: "Entertainment & Music",
        8: "Family & Relationships",
        9: "Politics & Government",
    },
    "AmazonReviewPolarity": {0: "Negative polarity", 1: "Positive polarity"},
    "AmazonReviewFull": {
        0: "score 1",
        1: "score 2",
        2: "score 3",
        3: "score 4",
        4: "score 5",
    },
}

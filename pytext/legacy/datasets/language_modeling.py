#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import io

from .. import data


class LanguageModelingDataset(data.Dataset):
    """Defines a dataset for language modeling."""

    def __init__(self, path, text_field, newline_eos=True, encoding="utf-8", **kwargs):
        """Create a LanguageModelingDataset given a path and a field.

        Args:
            path: Path to the data file.
            text_field: The field that will be used for text data.
            newline_eos: Whether to add an <eos> token for every newline in the
                data file. Default: True.
            encoding: The encoding of the file.
            kwargs: Passed to the constructor of
                data.Dataset.
        """
        fields = [("text", text_field)]
        text = []
        with io.open(path, encoding=encoding) as f:
            for line in f:
                text += text_field.preprocess(line)
                if newline_eos:
                    text.append(u"<eos>")

        examples = [data.Example.fromlist([text], fields)]
        super(LanguageModelingDataset, self).__init__(examples, fields, **kwargs)


class WikiText2(LanguageModelingDataset):

    urls = ["https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip"]
    name = "wikitext-2"
    dirname = "wikitext-2"

    @classmethod
    def splits(
        cls,
        text_field,
        root=".data",
        train="wiki.train.tokens",
        validation="wiki.valid.tokens",
        test="wiki.test.tokens",
        **kwargs
    ):
        """Create dataset objects for splits of the WikiText-2 dataset.

        This is the most flexible way to use the dataset.

        Args:
            text_field: The field that will be used for text data.
            root: The root directory that the dataset's zip archive will be
                expanded into; therefore the directory in whose wikitext-2
                subdirectory the data files will be stored.
            train: The filename of the train data. Default: 'wiki.train.tokens'.
            validation: The filename of the validation data, or None to not
                load the validation set. Default: 'wiki.valid.tokens'.
            test: The filename of the test data, or None to not load the test
                set. Default: 'wiki.test.tokens'.
        """
        return super(WikiText2, cls).splits(
            root=root,
            train=train,
            validation=validation,
            test=test,
            text_field=text_field,
            **kwargs
        )

    @classmethod
    def iters(
        cls, batch_size=32, bptt_len=35, device=0, root=".data", vectors=None, **kwargs
    ):
        """Create iterator objects for splits of the WikiText-2 dataset.

        This is the simplest way to use the dataset, and assumes common
        defaults for field, vocabulary, and iterator parameters.

        Args:
            batch_size: Batch size.
            bptt_len: Length of sequences for backpropagation through time.
            device: Device to create batches on. Use -1 for CPU and None for
                the currently active GPU device.
            root: The root directory that the dataset's zip archive will be
                expanded into; therefore the directory in whose wikitext-2
                subdirectory the data files will be stored.
            vectors: One of either the available pretrained vectors
                or custom pretrained vectors (see Vocab.load_vectors);
                or a list of aforementioned vectors
            kwargs: Passed to the splits method.
        """
        TEXT = data.Field()

        train, val, test = cls.splits(TEXT, root=root, **kwargs)

        TEXT.build_vocab(train, vectors=vectors)

        return data.BPTTIterator.splits(
            (train, val, test), batch_size=batch_size, bptt_len=bptt_len, device=device
        )


class WikiText103(LanguageModelingDataset):

    urls = [
        "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip"
    ]
    name = "wikitext-103"
    dirname = "wikitext-103"

    @classmethod
    def splits(
        cls,
        text_field,
        root=".data",
        train="wiki.train.tokens",
        validation="wiki.valid.tokens",
        test="wiki.test.tokens",
        **kwargs
    ):
        """Create dataset objects for splits of the WikiText-103 dataset.

        This is the most flexible way to use the dataset.

        Args:
            text_field: The field that will be used for text data.
            root: The root directory that the dataset's zip archive will be
                expanded into; therefore the directory in whose wikitext-103
                subdirectory the data files will be stored.
            train: The filename of the train data. Default: 'wiki.train.tokens'.
            validation: The filename of the validation data, or None to not
                load the validation set. Default: 'wiki.valid.tokens'.
            test: The filename of the test data, or None to not load the test
                set. Default: 'wiki.test.tokens'.
        """
        return super(WikiText103, cls).splits(
            root=root,
            train=train,
            validation=validation,
            test=test,
            text_field=text_field,
            **kwargs
        )

    @classmethod
    def iters(
        cls, batch_size=32, bptt_len=35, device=0, root=".data", vectors=None, **kwargs
    ):
        """Create iterator objects for splits of the WikiText-103 dataset.

        This is the simplest way to use the dataset, and assumes common
        defaults for field, vocabulary, and iterator parameters.

        Args:
            batch_size: Batch size.
            bptt_len: Length of sequences for backpropagation through time.
            device: Device to create batches on. Use -1 for CPU and None for
                the currently active GPU device.
            root: The root directory that the dataset's zip archive will be
                expanded into; therefore the directory in whose wikitext-2
                subdirectory the data files will be stored.
            vectors: One of either the available pretrained vectors
                or custom pretrained vectors (see Vocab.load_vectors);
                or a list of aforementioned vectors
            kwargs: Passed to the splits method.
        """
        TEXT = data.Field()

        train, val, test = cls.splits(TEXT, root=root, **kwargs)

        TEXT.build_vocab(train, vectors=vectors)

        return data.BPTTIterator.splits(
            (train, val, test), batch_size=batch_size, bptt_len=bptt_len, device=device
        )


class PennTreebank(LanguageModelingDataset):
    """The Penn Treebank dataset.
    A relatively small dataset originally created for POS tagging.

    References:
    Marcus, Mitchell P., Marcinkiewicz, Mary Ann & Santorini, Beatrice (1993).
    Building a Large Annotated Corpus of English: The Penn Treebank
    """

    urls = [
        "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt",
        "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt",
        "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt",
    ]
    name = "penn-treebank"
    dirname = ""

    @classmethod
    def splits(
        cls,
        text_field,
        root=".data",
        train="ptb.train.txt",
        validation="ptb.valid.txt",
        test="ptb.test.txt",
        **kwargs
    ):
        """Create dataset objects for splits of the Penn Treebank dataset.

        Args:
            text_field: The field that will be used for text data.
            root: The root directory where the data files will be stored.
            train: The filename of the train data. Default: 'ptb.train.txt'.
            validation: The filename of the validation data, or None to not
                load the validation set. Default: 'ptb.valid.txt'.
            test: The filename of the test data, or None to not load the test
                set. Default: 'ptb.test.txt'.
        """
        return super(PennTreebank, cls).splits(
            root=root,
            train=train,
            validation=validation,
            test=test,
            text_field=text_field,
            **kwargs
        )

    @classmethod
    def iters(
        cls, batch_size=32, bptt_len=35, device=0, root=".data", vectors=None, **kwargs
    ):
        """Create iterator objects for splits of the Penn Treebank dataset.

        This is the simplest way to use the dataset, and assumes common
        defaults for field, vocabulary, and iterator parameters.

        Args:
            batch_size: Batch size.
            bptt_len: Length of sequences for backpropagation through time.
            device: Device to create batches on. Use -1 for CPU and None for
                the currently active GPU device.
            root: The root directory where the data files will be stored.
            vectors: One of either the available pretrained vectors
                or custom pretrained vectors (see Vocab.load_vectors);
                or a list of aforementioned vectors
            kwargs: Passed to the splits method.
        """
        TEXT = data.Field()

        train, val, test = cls.splits(TEXT, root=root, **kwargs)

        TEXT.build_vocab(train, vectors=vectors)

        return data.BPTTIterator.splits(
            (train, val, test), batch_size=batch_size, bptt_len=bptt_len, device=device
        )

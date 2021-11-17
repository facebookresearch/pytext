#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import glob
import io
import os

from .. import data


class IMDB(data.Dataset):

    urls = ["http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"]
    name = "imdb"
    dirname = "aclImdb"

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path, text_field, label_field, **kwargs):
        """Create an IMDB dataset instance given a path and fields.

        Args:
            path: Path to the dataset's highest level directory
            text_field: The field that will be used for text data.
            label_field: The field that will be used for label data.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        fields = [("text", text_field), ("label", label_field)]
        examples = []

        for label in ["pos", "neg"]:
            for fname in glob.iglob(os.path.join(path, label, "*.txt")):
                with io.open(fname, "r", encoding="utf-8") as f:
                    text = f.readline()
                examples.append(data.Example.fromlist([text, label], fields))

        super(IMDB, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(
        cls, text_field, label_field, root=".data", train="train", test="test", **kwargs
    ):
        """Create dataset objects for splits of the IMDB dataset.

        Args:
            text_field: The field that will be used for the sentence.
            label_field: The field that will be used for label data.
            root: Root dataset storage directory. Default is '.data'.
            train: The directory that contains the training examples
            test: The directory that contains the test examples
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        return super(IMDB, cls).splits(
            root=root,
            text_field=text_field,
            label_field=label_field,
            train=train,
            validation=None,
            test=test,
            **kwargs
        )

    @classmethod
    def iters(cls, batch_size=32, device=0, root=".data", vectors=None, **kwargs):
        """Create iterator objects for splits of the IMDB dataset.

        Args:
            batch_size: Batch_size
            device: Device to create batches on. Use - 1 for CPU and None for
                the currently active GPU device.
            root: The root directory that contains the imdb dataset subdirectory
            vectors: one of the available pretrained vectors or a list with each
                element one of the available pretrained vectors (see Vocab.load_vectors)

            Remaining keyword arguments: Passed to the splits method.
        """
        TEXT = data.Field()
        LABEL = data.Field(sequential=False)

        train, test = cls.splits(TEXT, LABEL, root=root, **kwargs)

        TEXT.build_vocab(train, vectors=vectors)
        LABEL.build_vocab(train)

        return data.BucketIterator.splits(
            (train, test), batch_size=batch_size, device=device
        )

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import random

from .. import data


class SequenceTaggingDataset(data.Dataset):
    """Defines a dataset for sequence tagging. Examples in this dataset
    contain paired lists -- paired list of words and tags.

    For example, in the case of part-of-speech tagging, an example is of the
    form
    [I, love, PyTorch, .] paired with [PRON, VERB, PROPN, PUNCT]

    See torchtext/test/sequence_tagging.py on how to use this class.
    """

    @staticmethod
    def sort_key(example):
        for attr in dir(example):
            if not callable(getattr(example, attr)) and not attr.startswith("__"):
                return len(getattr(example, attr))
        return 0

    def __init__(self, path, fields, encoding="utf-8", separator="\t", **kwargs):
        examples = []
        columns = []

        with open(path, encoding=encoding) as input_file:
            for line in input_file:
                line = line.strip()
                if line == "":
                    if columns:
                        examples.append(data.Example.fromlist(columns, fields))
                    columns = []
                else:
                    for i, column in enumerate(line.split(separator)):
                        if len(columns) < i + 1:
                            columns.append([])
                        columns[i].append(column)

            if columns:
                examples.append(data.Example.fromlist(columns, fields))
        super(SequenceTaggingDataset, self).__init__(examples, fields, **kwargs)


class UDPOS(SequenceTaggingDataset):

    # Universal Dependencies English Web Treebank.
    # Download original at http://universaldependencies.org/
    # License: http://creativecommons.org/licenses/by-sa/4.0/
    urls = ["https://bitbucket.org/sivareddyg/public/downloads/en-ud-v2.zip"]
    dirname = "en-ud-v2"
    name = "udpos"

    @classmethod
    def splits(
        cls,
        fields,
        root=".data",
        train="en-ud-tag.v2.train.txt",
        validation="en-ud-tag.v2.dev.txt",
        test="en-ud-tag.v2.test.txt",
        **kwargs
    ):
        """Downloads and loads the Universal Dependencies Version 2 POS Tagged
        data.
        """

        return super(UDPOS, cls).splits(
            fields=fields,
            root=root,
            train=train,
            validation=validation,
            test=test,
            **kwargs
        )


class CoNLL2000Chunking(SequenceTaggingDataset):
    # CoNLL 2000 Chunking Dataset
    # https://www.clips.uantwerpen.be/conll2000/chunking/
    urls = [
        "https://www.clips.uantwerpen.be/conll2000/chunking/train.txt.gz",
        "https://www.clips.uantwerpen.be/conll2000/chunking/test.txt.gz",
    ]
    dirname = ""
    name = "conll2000"

    @classmethod
    def splits(
        cls,
        fields,
        root=".data",
        train="train.txt",
        test="test.txt",
        validation_frac=0.1,
        **kwargs
    ):
        """Downloads and loads the CoNLL 2000 Chunking dataset.
        NOTE: There is only a train and test dataset so we use 10% of the train set as validation
        """

        train, test = super(CoNLL2000Chunking, cls).splits(
            fields=fields, root=root, train=train, test=test, separator=" ", **kwargs
        )

        # HACK: Saving the sort key function as the split() call removes it
        sort_key = train.sort_key

        # Now split the train set
        # Force a random seed to make the split deterministic
        random.seed(0)
        train, val = train.split(1 - validation_frac, random_state=random.getstate())
        # Reset the seed
        random.seed()

        # HACK: Set the sort key
        train.sort_key = sort_key
        val.sort_key = sort_key

        return train, val, test

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os

from .. import data


class TREC(data.Dataset):

    urls = [
        "http://cogcomp.org/Data/QA/QC/train_5500.label",
        "http://cogcomp.org/Data/QA/QC/TREC_10.label",
    ]
    name = "trec"
    dirname = ""

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path, text_field, label_field, fine_grained=False, **kwargs):
        """Create an TREC dataset instance given a path and fields.

        Args:
            path: Path to the data file.
            text_field: The field that will be used for text data.
            label_field: The field that will be used for label data.
            fine_grained: Whether to use the fine-grained (50-class) version of TREC
                or the coarse grained (6-class) version.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        fields = [("text", text_field), ("label", label_field)]
        examples = []

        def get_label_str(label):
            return label.split(":")[0] if not fine_grained else label

        label_field.preprocessing = data.Pipeline(get_label_str)

        for line in open(os.path.expanduser(path), "rb"):
            # there is one non-ASCII byte: sisterBADBYTEcity; replaced with space
            label, _, text = line.replace(b"\xf0", b" ").decode().partition(" ")
            examples.append(data.Example.fromlist([text, label], fields))

        super(TREC, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(
        cls,
        text_field,
        label_field,
        root=".data",
        train="train_5500.label",
        test="TREC_10.label",
        **kwargs
    ):
        """Create dataset objects for splits of the TREC dataset.

        Args:
            text_field: The field that will be used for the sentence.
            label_field: The field that will be used for label data.
            root: Root dataset storage directory. Default is '.data'.
            train: The filename of the train data. Default: 'train_5500.label'.
            test: The filename of the test data, or None to not load the test
                set. Default: 'TREC_10.label'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        return super(TREC, cls).splits(
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
        """Create iterator objects for splits of the TREC dataset.

        Args:
            batch_size: Batch_size
            device: Device to create batches on. Use - 1 for CPU and None for
                the currently active GPU device.
            root: The root directory that contains the trec dataset subdirectory
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

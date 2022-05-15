#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
from io import open

import torch

from ..data import Dataset, Example, Field, Iterator


class BABI20Field(Field):
    def __init__(self, memory_size, **kwargs):
        super(BABI20Field, self).__init__(**kwargs)
        self.memory_size = memory_size
        self.unk_token = None
        self.batch_first = True

    def preprocess(self, x):
        if isinstance(x, list):
            return [super(BABI20Field, self).preprocess(s) for s in x]
        else:
            return super(BABI20Field, self).preprocess(x)

    def pad(self, minibatch):
        if isinstance(minibatch[0][0], list):
            self.fix_length = max(max(len(x) for x in ex) for ex in minibatch)
            padded = []
            for ex in minibatch:
                # sentences are indexed in reverse order and truncated to memory_size
                nex = ex[::-1][: self.memory_size]
                padded.append(
                    super(BABI20Field, self).pad(nex)
                    + [[self.pad_token] * self.fix_length]
                    * (self.memory_size - len(nex))
                )
            self.fix_length = None
            return padded
        else:
            return super(BABI20Field, self).pad(minibatch)

    def numericalize(self, arr, device=None):
        if isinstance(arr[0][0], list):
            tmp = [
                super(BABI20Field, self).numericalize(x, device=device).data
                for x in arr
            ]
            arr = torch.stack(tmp)
            if self.sequential:
                arr = arr.contiguous()
            return arr
        else:
            return super(BABI20Field, self).numericalize(arr, device=device)


class BABI20(Dataset):
    urls = ["http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz"]
    name = ""
    dirname = ""

    def __init__(self, path, text_field, only_supporting=False, **kwargs):
        fields = [("story", text_field), ("query", text_field), ("answer", text_field)]
        self.sort_key = lambda x: len(x.query)

        with open(path, "r", encoding="utf-8") as f:
            triplets = self._parse(f, only_supporting)
            examples = [Example.fromlist(triplet, fields) for triplet in triplets]

        super(BABI20, self).__init__(examples, fields, **kwargs)

    @staticmethod
    def _parse(file, only_supporting):
        data, story = [], []
        for line in file:
            tid, text = line.rstrip("\n").split(" ", 1)
            if tid == "1":
                story = []
            # sentence
            if text.endswith("."):
                story.append(text[:-1])
            # question
            else:
                # remove any leading or trailing whitespace after splitting
                query, answer, supporting = (x.strip() for x in text.split("\t"))
                if only_supporting:
                    substory = [story[int(i) - 1] for i in supporting.split()]
                else:
                    substory = [x for x in story if x]
                data.append((substory, query[:-1], answer))  # remove '?'
                story.append("")
        return data

    @classmethod
    def splits(
        cls,
        text_field,
        path=None,
        root=".data",
        task=1,
        joint=False,
        tenK=False,
        only_supporting=False,
        train=None,
        validation=None,
        test=None,
        **kwargs
    ):
        assert isinstance(task, int) and 1 <= task <= 20
        if tenK:
            cls.dirname = os.path.join("tasks_1-20_v1-2", "en-valid-10k")
        else:
            cls.dirname = os.path.join("tasks_1-20_v1-2", "en-valid")
        if path is None:
            path = cls.download(root)
        if train is None:
            if joint:  # put all tasks together for joint learning
                train = "all_train.txt"
                if not os.path.isfile(os.path.join(path, train)):
                    with open(os.path.join(path, train), "w") as tf:
                        for task in range(1, 21):
                            with open(
                                os.path.join(path, "qa" + str(task) + "_train.txt")
                            ) as f:
                                tf.write(f.read())
            else:
                train = "qa" + str(task) + "_train.txt"
        if validation is None:
            if joint:  # put all tasks together for joint learning
                validation = "all_valid.txt"
                if not os.path.isfile(os.path.join(path, validation)):
                    with open(os.path.join(path, validation), "w") as tf:
                        for task in range(1, 21):
                            with open(
                                os.path.join(path, "qa" + str(task) + "_valid.txt")
                            ) as f:
                                tf.write(f.read())
            else:
                validation = "qa" + str(task) + "_valid.txt"
        if test is None:
            test = "qa" + str(task) + "_test.txt"
        return super(BABI20, cls).splits(
            path=path,
            root=root,
            text_field=text_field,
            train=train,
            validation=validation,
            test=test,
            **kwargs
        )

    @classmethod
    def iters(
        cls,
        batch_size=32,
        root=".data",
        memory_size=50,
        task=1,
        joint=False,
        tenK=False,
        only_supporting=False,
        sort=False,
        shuffle=False,
        device=None,
        **kwargs
    ):
        text = BABI20Field(memory_size)
        train, val, test = BABI20.splits(
            text,
            root=root,
            task=task,
            joint=joint,
            tenK=tenK,
            only_supporting=only_supporting,
            **kwargs
        )
        text.build_vocab(train)
        return Iterator.splits(
            (train, val, test),
            batch_size=batch_size,
            sort=sort,
            shuffle=shuffle,
            device=device,
        )

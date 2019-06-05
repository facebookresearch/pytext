#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import json
from typing import List, Optional

from pytext.data.sources.data_source import DataSource, generator_property
from pytext.data.sources.tsv import TSVDataSource


def unflatten(fname, ignore_impossible):
    if not fname:
        return
    with open(fname) as file:
        dump = json.load(file)

    for article in dump["data"]:
        for paragraph in article["paragraphs"]:
            doc = paragraph["context"]
            for question in paragraph["qas"]:
                has_answer = not question.get("is_impossible", False)
                if has_answer or not ignore_impossible:
                    answers = (
                        question["answers"]
                        if has_answer
                        else question["plausible_answers"]
                    )
                    yield {
                        "doc": doc,
                        "question": question["question"],
                        "answers": [answer["text"] for answer in answers],
                        "answer_starts": [int(ans["answer_start"]) for ans in answers],
                        "has_answer": str(has_answer),
                    }


class SquadDataSource(DataSource):
    """
    Download data from https://rajpurkar.github.io/SQuAD-explorer/
    Will return tuples of (doc, question, answer, answer_start, has_answer)
    """

    class Config(DataSource.Config):
        train_filename: Optional[str] = "train-v2.0.json"
        test_filename: Optional[str] = "dev-v2.0.json"
        eval_filename: Optional[str] = "dev-v2.0.json"
        ignore_impossible: bool = True

    @classmethod
    def from_config(cls, config: Config, schema=None):
        return cls(
            config.train_filename,
            config.test_filename,
            config.eval_filename,
            config.ignore_impossible,
        )

    def __init__(
        self,
        train_filename=None,
        test_filename=None,
        eval_filename=None,
        ignore_impossible=Config.ignore_impossible,
    ):
        schema = {
            "doc": str,
            "question": str,
            "answers": List[str],
            "answer_starts": List[int],
            "answer_ends": List[int],
            "has_answer": str,
        }
        super().__init__(schema)
        self.train_filename = train_filename
        self.test_filename = test_filename
        self.eval_filename = eval_filename
        self.ignore_impossible = ignore_impossible

    @generator_property
    def train(self):
        return unflatten(self.train_filename, self.ignore_impossible)

    @generator_property
    def test(self):
        return unflatten(self.test_filename, self.ignore_impossible)

    @generator_property
    def eval(self):
        return unflatten(self.eval_filename, self.ignore_impossible)


class SquadTSVDataSource(TSVDataSource):
    """
    Squad-like data passed in TSV format.
    Will return tuples of (doc, question, answer, answer_start, has_answer)
    """

    class Config(TSVDataSource.Config):
        field_names: List[str] = [
            "doc",
            "question",
            "answers",
            "answer_starts",
            "has_answer",
        ]

    def __init__(self, **kwargs):
        kwargs["schema"] = {
            "doc": str,
            "question": str,
            "answers": List[str],
            "answer_starts": List[int],
            "has_answer": str,
        }
        super().__init__(**kwargs)

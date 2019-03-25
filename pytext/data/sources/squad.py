#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import json
from typing import List, Optional

from pytext.data.sources.data_source import DataSource, generator_property


def unflatten(fname, ignore_impossible):
    if not fname:
        return
    with open(fname) as file:
        dump = json.load(file)
    for article in dump["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for question in paragraph["qas"]:
                label = not question["is_impossible"]
                if label or not ignore_impossible:
                    answers = (
                        question["answers"] if label else question["plausible_answers"]
                    )
                    yield {
                        "context": context,
                        "question": question["question"],
                        "answers": [answer["text"] for answer in answers],
                        "answer_starts": [int(ans["answer_start"]) for ans in answers],
                        "label": label,
                    }


class SquadDataSource(DataSource):
    """Download data from https://rajpurkar.github.io/SQuAD-explorer/
       Will return tuples of (context, question, answer, answer_start, label, weight)
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
            "context": str,
            "question": str,
            "answers": List[str],
            "answer_starts": List[int],
            "answer_ends": List[int],
            "label": str,
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

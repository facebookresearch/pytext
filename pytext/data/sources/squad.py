#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import json
from typing import List, Optional

from pytext.data.sources.data_source import DataSource, generator_property
from pytext.data.sources.tsv import TSV, SafeFileWrapper


def get_json_iter(fname, ignore_impossible):
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


def get_tsv_iter(file_path, field_names, delimiter):
    for row in iter(
        TSV(
            SafeFileWrapper(file_path, encoding="utf-8", errors="replace"),
            field_names=field_names,
            delimiter=delimiter,
        )
    ):
        yield {
            "doc": row["doc"],
            "question": row["question"],
            "answers": json.loads(row["answers"]),
            "answer_starts": json.loads(row["answer_starts"]),
            "has_answer": json.loads(row["has_answer"]),
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
        is_json: bool = True
        field_names: List[str] = [
            "doc",
            "question",
            "answers",
            "answer_starts",
            "has_answer",
        ]
        delimiter: str = "\t"

    @classmethod
    def from_config(cls, config: Config, schema=None):
        return cls(
            config.train_filename,
            config.test_filename,
            config.eval_filename,
            config.ignore_impossible,
            config.is_json,
            config.field_names,
            config.delimiter,
        )

    def __init__(
        self,
        train_filename=None,
        test_filename=None,
        eval_filename=None,
        ignore_impossible=Config.ignore_impossible,
        is_json=Config.is_json,
        field_names=Config.field_names,
        delimiter=Config.delimiter,
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
        self.is_json = is_json
        self.field_names = field_names
        self.delimiter = delimiter

    @generator_property
    def train(self):
        return (
            get_json_iter(self.train_filename, self.ignore_impossible)
            if self.is_json
            else get_tsv_iter(self.train_filename, self.field_names, self.delimiter)
        )

    @generator_property
    def test(self):
        return (
            get_json_iter(self.test_filename, self.ignore_impossible)
            if self.is_json
            else get_tsv_iter(self.train_filename, self.field_names, self.delimiter)
        )

    @generator_property
    def eval(self):
        return (
            get_json_iter(self.eval_filename, self.ignore_impossible)
            if self.is_json
            else get_tsv_iter(self.train_filename, self.field_names, self.delimiter)
        )
